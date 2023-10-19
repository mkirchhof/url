#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
Modification copyright 2023 Michael Kirchhof
"""
import argparse
import logging
import os
import time
import sys
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset, \
    ImagenetTransform, prepare_n_crop_transform
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy, \
    CrossEntropy, ExpectedLikelihoodKernel, MCInfoNCE, InfoNCE, HedgedInstance, \
    LossPrediction, NonIsotropicVMF
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler, str2bool
from validate import validate_one_epoch, validate_on_downstream_datasets

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR', default="./data/ImageNet2012",
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='torch/imagenet',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--data-dir-eval', metavar='DIR', default=None,
                    help='path to root dir where eval dataset is stored. If None, use --data-dir')
parser.add_argument('--dataset_eval', metavar='NAME', default='torch/imagenet',
                    help='Which dataset to evaluate on (usually the same or a soft label variant of --dataset)')
parser.add_argument('--data-dir-downstream', metavar='DIR', default="./data",
                    help='path to root dir where downstream datasets are stored. If None, use --data-dir')
parser.add_argument('--dataset-downstream', nargs='+', default=["repr/cub", "repr/cars", "repr/sop"],
                    type=str, help='list dataset type + name ("<type>/<name>") for the zero-shot downstream metrics. To skip downstream evaluation, just provide the same as for dataset_eval.')
parser.add_argument('--further-dataset-downstream', nargs='+', default=[],
                    type=str, help='list dataset type + name ("<type>/<name>") to test performance on some more datasets youre interested in.')
parser.add_argument('--max-num-samples', default=100000, type=int,
                   help='Maximum number of samples in concatenated ID + OOD dataset (default: 50000)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (train/validation/test/trainontest/testontest). The last two are for experiments where we train on the test classes.')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (train/validation/test/trainontest/testontest). The last two are for experiments where we train on the test classes.')
group.add_argument('--test-split', metavar='NAME', default='test',
                   help='dataset test split (train/validation/test/trainontest/testontest). The last two are for experiments where we train on the test classes.')
group.add_argument('--dataset-download', type=str2bool, default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
group.add_argument('--n_few_shot', default=None, type=int,
                   help="If not None, to how many samples per class to restrict the train dataset to. Currently only implemented for repr/ datasets.")

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--loss', default="cross-entropy", type=str,
                   help="Loss function to use (cross-entropy, elk, nivmf, hib, vmf, mcinfonce, infonce, losspred)")
group.add_argument('--ssl', default=False, type=str2bool, help="Whether to train using self-supervised learning.")
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--num-heads', default=1, type=int,
                   help='Number of heads in a shallow ensemble (default: 1 -- no ensembling)')
group.add_argument('--lambda-value', default=1.0, type=float,
                   help='Trade-off hyperparameter between task loss and uncertainty loss when using "pred-net" (default: 1.0)')
group.add_argument('--rank_V', default=50, type=int,
                   help='Rank of V in HET-XL (default: 50)')
group.add_argument('--c-mult', default=0.1, type=float,
                   help='Multiplier of C initializer in HET-XL (default: 0.1)')
group.add_argument('--gp-input-normalization', default=False, type=str2bool,
                   help='Whether to use GP input normalization in SNGP (default: False)')
group.add_argument('--gp-cov-discount-factor', default=0.999, type=float,
                   help='GP covariance matrix discount factor in SNGP (default: 0.999)')
group.add_argument('--use-spec-norm', default=False, type=str2bool,
                   help='Whether to use spectral normalization in SNGP (default: False)')
group.add_argument('--spec-norm-bound', default=6, type=float,
                   help='Spectral normalization multiplier in SNGP (default: 6)')
group.add_argument('--mc_samples', default=16, type=int,
                   help="Number of MC samples used to calculate probabilistic losses.")
parser.add_argument('--unc-module', metavar='NAME', default='pred-net',
                    help='What to use to estimate aleatoric uncertainty (none, class-entropy, jsd, embed-norm, pred-net, hetxl-det)')
parser.add_argument('--unc_start_value', default=0.001, type=float,
                    help='Which (average) uncertainty value to start from (higher=more variance). 0 to ignore.')
parser.add_argument('--unc_width', default=1024, type=int,
                    help='Width of the pred-net of the unc-module')
group.add_argument('--pretrained', type=str2bool, default=True,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--freeze_backbone', type=str2bool, default=False,
                   help='Whether to freeze the ResNet/ViT/... and only train the uncertainty module.')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', type=str2bool, default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('--accumulation_steps', type=int, default=16,
                   help="How many batches to accumulate before making an optimizer step (to simulate bigger batchsize)")
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', type=str2bool, default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-checkpointing', type=str2bool, default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, type=str2bool,
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--inv_temp', default=28, type=float,
                   help="= 1/temperature for the loss")
group.add_argument('--hib_add_const', default=0, type=float,
                   help="Additive constant in the HIB loss")

scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', type=str2bool, default=False,
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, type=str2bool,
                             help="Enable AOT Autograd support.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='lamb', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', type=str2bool, default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.001, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-4, metavar='LR',
                   help='warmup learning rate (default: 1e-4)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=32, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', type=str2bool, default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', type=str2bool, default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', type=str2bool, default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', type=str2bool, default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', type=str2bool, default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', type=str2bool, default=False,
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', type=str2bool, default=False,
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', type=str2bool, default=False,
                   help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', type=str2bool, default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='decay factor for model weights moving average (default: 0.9998)')

# For evaluating uncertainty
parser.add_argument('--crop_min', default=0.1, type=float, help="Minimal crop pct for deteriorating an image.")
parser.add_argument('--crop_max', default=0.75, type=float, help="Maximum crop pct for deteriorating an image.")

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--save-images', type=str2bool, default=False,
                   help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', type=str2bool, default=True,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', type=str2bool, default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', type=str2bool, default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', type=str2bool, default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='avg_downstream_auroc_correct', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', type=str2bool, default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', type=str2bool, default=True,
                   help='log training and validation metrics to wandb')
group.add_argument('--wandb-key', type=str, default="",
                   help="Your wandb API key")


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Include this if you are too lazy to specify number of classes in few-shot
    # if args.n_few_shot is not None:
    #     args.dataset_eval = args.dataset
    #     args.dataset_downstream = [args.dataset]
    #     args.num_classes = {"repr/cub": 100, "repr/cars": 98, "repr/sop": 11318}[args.dataset]

    if args.data_dir_eval is None:
        args.data_dir_eval = args.data_dir
    if args.data_dir_downstream is None:
        args.data_dir_downstream = args.data_dir

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    time_start_setup = datetime.now()
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    else:
        _logger.info("CUDA is not available.")

    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    #if args.distributed:
    #    _logger.info(f'Training in distributed mode with multiple processes, 1 device per process. Process {args.rank}, total {args.world_size}, device {args.device}.')
    #else:
    #    _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        unc_module=args.unc_module,
        unc_width=args.unc_width,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        num_heads=args.num_heads,
        rank_V=args.rank_V,
        c_mult=args.c_mult,
        gp_input_normalization=args.gp_input_normalization,
        gp_cov_discount_factor=args.gp_cov_discount_factor,
        use_spec_norm=args.use_spec_norm,
        spec_norm_bound=args.spec_norm_bound,
        **args.model_kwargs
    )

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        n_params = sum([m.numel() for m in model.parameters()])
        _logger.info(f'Model {safe_model_name(args.model)} created, param count: {n_params}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # freeze model backbone
    if args.freeze_backbone:
        if args.unc_module == "pred-net":
            model.freeze_backbone()
        else:
            _logger.warning("Freezing backbone is forbidden if --unc_module != pred-net. Continuing without frozen backbone.")

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        # FIXME dynamo might need move below DDP wrapping? TBD
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.accumulation_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) and global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda':
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args)
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        n_few_shot=args.n_few_shot,
    )

    # Create the validation datasets
    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)

    dataset_locations = {}
    dataset_locations[args.dataset_eval] = (args.data_dir_eval, eval_workers)
    for dataset in args.dataset_downstream:
        dataset_locations[dataset] = (
        args.data_dir_downstream, 1)

    datasets_eval = {}
    if args.val_split is not None and args.val_split != "none":
        for name, (location, num_workers) in dataset_locations.items():
            dataset = create_dataset(
                root=location,
                name=name,
                split=args.val_split,
                download=args.dataset_download,
                class_map=args.class_map,
                batch_size=args.batch_size,
                is_training=False
            )
            datasets_eval[name] = (dataset, num_workers)

    # Create the test datasets
    dataset_locations_test = {}
    for dataset in args.dataset_downstream:
        dataset_locations_test[dataset] = (
            args.data_dir_downstream, 1)

    datasets_test = {}
    datasets_test2 = {}
    if args.test_split is not None and args.test_split != "none":
        for name, (location, num_workers) in dataset_locations_test.items():
            dataset = create_dataset(
                root=location,
                name=name,
                split=args.test_split,
                download=args.dataset_download,
                class_map=args.class_map,
                batch_size=args.batch_size,
                is_training=False
            )
            datasets_test[name] = (dataset, num_workers)

        # Create the test datasets
        dataset_locations_test2 = {}
        for dataset in args.further_dataset_downstream:
            dataset_locations_test2[dataset] = (
                args.data_dir_downstream, 1)

        for name, (location, num_workers) in dataset_locations_test2.items():
            dataset = create_dataset(
                root=location,
                name=name,
                split=args.test_split,
                download=args.dataset_download,
                class_map=args.class_map,
                batch_size=args.batch_size,
                is_training=False
            )
            datasets_test2[name] = (dataset, num_workers)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    transform = None
    if args.ssl:
        transform = prepare_n_crop_transform([ImagenetTransform(
            interpolation=train_interpolation, crop_size=data_config['input_size'][1])],
            num_crops_per_aug=[2])  # This might need to be 1 or 1 1 to get positive pairs
        # Note: I'm also hoping that solo learn doesn't need the dataset_with_index.
        # Otherwise, we'll need to adapt the create_loader function accordingly. See prepare_datasets() in solo learn
    loader_train = create_loader(
        dataset_train,
        transform=transform,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers if not args.n_few_shot else 1,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding
    )

    dataloaders_eval = {}
    for name, (dataset, num_workers) in datasets_eval.items():
        dataloaders_eval[name] = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=num_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            device=device
        )
    loader_eval_upstream = dataloaders_eval.pop(args.dataset_eval)
    loaders_eval_downstream = dataloaders_eval

    dataloaders_test = {}
    for name, (dataset, num_workers) in datasets_test.items():
        dataloaders_test[name] = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=num_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            device=device
        )
    loaders_test = dataloaders_test

    dataloaders_test2 = {}
    for name, (dataset, num_workers) in datasets_test2.items():
        dataloaders_test2[name] = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=num_workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            device=device
        )
    loaders_test2 = dataloaders_test2

    # Initialize uncertainty module
    if args.initial_checkpoint == "" and args.unc_start_value > 0:
        with amp_autocast():
            model.initialize_avg_uncertainty(loader_train, args.unc_start_value)

    # setup loss function
    if args.loss == "cross-entropy":
        if args.jsd_loss:
            assert num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
        elif mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif args.smoothing:
            if args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            train_loss_fn = CrossEntropy()
        validate_loss_fn = CrossEntropy()
    elif args.loss == "elk":
        train_loss_fn = ExpectedLikelihoodKernel(inv_temp=args.inv_temp)
        validate_loss_fn = ExpectedLikelihoodKernel(inv_temp=args.inv_temp)
    elif args.loss == "nivmf":
        train_loss_fn = NonIsotropicVMF(n_classes=args.num_classes, embed_dim=model.num_features, inv_temp=args.inv_temp, n_samples=args.mc_samples)
        validate_loss_fn = NonIsotropicVMF(n_classes=args.num_classes, embed_dim=model.num_features, inv_temp=args.inv_temp, n_samples=args.mc_samples)
    elif args.loss == "mcinfonce":
        train_loss_fn = MCInfoNCE(kappa_init=args.inv_temp, n_samples=args.mc_samples)
        validate_loss_fn = CrossEntropy()
    elif args.loss == "infonce":
        train_loss_fn = InfoNCE(kappa_init=args.inv_temp)
        validate_loss_fn = CrossEntropy()
    elif args.loss == "hib":
        train_loss_fn = HedgedInstance(kappa_init=args.inv_temp, b=args.hib_add_const, n_samples=args.mc_samples)
        validate_loss_fn = CrossEntropy()
    elif args.loss == "losspred":
        train_loss_fn = LossPrediction(lambda_=args.lambda_value, ignore_ce_loss=args.freeze_backbone)
        validate_loss_fn = CrossEntropy()
    else:
        raise NotImplementedError(f"--loss {args.loss} is not implemented.")
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = validate_loss_fn.to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_test_metrics = None
    best_test_metrics2 = None
    best_eval_metrics = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S%f"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
    best_eval_metric = 0
    best_test_metrics = None

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb.init(project="large", name=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    time_start_epoch = datetime.now()
    _logger.info(f"Setup took {(time_start_epoch - time_start_setup).total_seconds()} seconds")
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            if args.lr > 0:
                train_metrics = train_one_epoch(
                    epoch,
                    model,
                    loader_train,
                    optimizer,
                    train_loss_fn,
                    args,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    mixup_fn=mixup_fn
                )
            else:
                _logger.info("Learning rate is 0, skipping training epoch.")
                train_metrics = None

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate_one_epoch(
                model,
                loader_eval_upstream,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
                is_upstream=True,
                dataset_name=args.dataset_eval,
                log_suffix=f"Val {args.dataset_eval}",
                tempfolder=output_dir
            )
            _logger.info(
                'Upstream * Acc@1 {:.3f} R@1 {:.3f} AUROC-R@1 {:.3f} croppedHasBiggerUnc {:.3f}'.format(
                    eval_metrics['top1'], eval_metrics['r1'], eval_metrics["auroc_correct"], eval_metrics['croppedHasBiggerUnc']))

            if len(loaders_eval_downstream) > 0:
                eval_metrics.update(validate_on_downstream_datasets(
                    model,
                    loaders_eval_downstream,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    tempfolder=output_dir,
                    log_suffix="Val"
                ))

                _logger.info(
                    'Downstream * R@1 {:.3f} AUROC-correctness {:.3f} croppedHasBiggerUnc {:.3f}'.format(
                        eval_metrics["avg_downstream_r1"], eval_metrics["avg_downstream_auroc_correct"], eval_metrics["avg_downstream_croppedHasBiggerUnc"]))

            is_new_best = (epoch == start_epoch or
                        (decreasing and eval_metrics[eval_metric] < best_eval_metric) or
                        ((not decreasing) and eval_metrics[eval_metric] > best_eval_metric))
            if is_new_best:
                best_eval_metric = eval_metrics[eval_metric]
                best_eval_metrics = eval_metrics

                # Only for best models, track the test scores
                if len(loaders_test) > 0:
                    best_test_metrics = validate_on_downstream_datasets(
                        model,
                        loaders_test,
                        validate_loss_fn,
                        args,
                        amp_autocast=amp_autocast,
                        tempfolder=output_dir,
                        log_suffix="Test"
                    )

                if len(loaders_test2) > 0:
                    best_test_metrics2 = validate_on_downstream_datasets(
                        model,
                        loaders_test2,
                        validate_loss_fn,
                        args,
                        amp_autocast=amp_autocast,
                        tempfolder=output_dir,
                        log_suffix="Further test"
                    )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate_one_epoch(
                    model_ema.module,
                    loader_eval_upstream,
                    validate_loss_fn,
                    args,
                    amp_autocast=amp_autocast,
                    log_suffix=f'Val (EMA) {args.dataset_eval}',
                    is_upstream=True,
                    dataset_name=args.dataset,
                    tempfolder=output_dir
                )

                if len(loaders_eval_downstream) > 0:
                    ema_eval_metrics.update(validate_on_downstream_datasets(
                        model,
                        loaders_eval_downstream,
                        validate_loss_fn,
                        args,
                        amp_autocast=amp_autocast,
                        tempfolder=output_dir,
                        log_suffix='Val (EMA)'
                    ))

                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    best_eval_metrics=best_eval_metrics,
                    best_test_metrics=best_test_metrics,
                    best_test_metrics2=best_test_metrics2,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            time_end_epoch = datetime.now()
            _logger.info(f"Epoch {epoch} took {(time_end_epoch - time_start_epoch).total_seconds()} seconds")
            time_start_epoch = time_end_epoch

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    accuracy_m = utils.AverageMeter()

    model.train()
    optimizer.zero_grad()

    end = time.time()
    num_batches_per_epoch = len(loader)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx

        # If we get soft labels, sample a hard label from them
        if len(target.shape) == 2:
            target = (target[:,1:]).float().multinomial(1).squeeze(-1)

        #data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output, unc, features = model(input)
            loss = loss_fn(output, unc, target, features, model.get_classifier()) / args.accumulation_steps

        if not args.distributed:
            losses_m.update(loss.item() * args.accumulation_steps, input.size(0))

        do_optim_step = (batch_idx + 1) % args.accumulation_steps == 0 # Do not include an "or last_batch" here to avoid
                                                                       # incomplete last batch if we accumulate gradients
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order,
                do_optim_step=do_optim_step
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode
                )
            if do_optim_step:
                optimizer.step()

        #grad_norm_m.update(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=torch.inf))

        if do_optim_step:
            optimizer.zero_grad()

        if model_ema is not None:
            model_ema.update(model)

        # Calculate accuracy
        pred_correct = utils.is_pred_correct(output.detach(), target).float().squeeze()
        accuracy_m.update(pred_correct.mean().item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if utils.is_primary(args):
                _logger.info(
                    'Train: {} [{:>4d}/{}]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Accuracy: {acc.val:.3f} ({acc.avg:.3f})  '
                    'LR: {lr:.3e}  '
                    'Remaining time: {mins_left:>4d}min'.format(
                        epoch,
                        batch_idx, len(loader),
                        loss=losses_m,
                        acc=accuracy_m,
                        lr=lr,
                        mins_left=round((len(loader) - batch_idx) * batch_time_m.avg / 60))
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


if __name__ == '__main__':
    main()

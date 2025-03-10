#!/usr/bin/env python3

"""
This file creates a dataloader and a (frozen) model, then caches the model embeddings
into a h5py file. It simulates either validation (one epoch), or training (multiple
epochs with random augmentations). You can then load the h5py file as a dataloader,
speeding up training by ~20x and making it possible to finetune even without GPU.

Since this caches model embeddings, it only makes sense for frozen model backbones.
But you could adjust it to cache the randomly augmented images only instead.

Example for a train dataset:
python create_cached_dataset.py --data-dir /path/to/data/folder --dataset torch/imagenet --batch-size 128
--split train --model resnet50 --epochs 100 --output /path/to/output/folder

Example for a test dataset (disabling all augmentations and shuffling):
python create_cached_dataset.py --data-dir /path/to/data/folder --dataset torch/imagenet --batch-size 128
--split val --is_training false --model resnet50 --epochs 1 --output /path/to/output/folder

Based on 2020's timm library by Ross Wightman (https://github.com/rwightman)
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
import h5py

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset, \
    ImagenetTransform, prepare_n_crop_transform
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.utils import str2bool

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

# Dataset parameters
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('--data-dir', metavar='DIR', default="./data/ImageNet2012",
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='torch/imagenet',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--split', metavar='NAME', default='train',
                   help='dataset split (train/validation/test/trainontest/testontest). The last two are for experiments where we train on the test classes.')
group.add_argument('--is_training', type=str2bool, default=None,
                   help="If false, turns off shuffling and random augmentations. Default: '--split == train'")
group.add_argument('--dataset-download', type=str2bool, default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
group.add_argument('--n_few_shot', default=None, type=int,
                   help="If not None, to how many samples per class to restrict the train dataset to. Currently only implemented for repr/ datasets.")

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--ssl', default=False, type=str2bool, help="Whether to train using self-supervised learning.")
group.add_argument('--num-heads', default=1, type=int,
                   help='Number of heads in a shallow ensemble (default: 1 -- no ensembling)')
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
parser.add_argument('--unc-module', metavar='NAME', default='pred-net',
                    help='What to use to estimate aleatoric uncertainty (none, class-entropy, jsd, embed-norm, pred-net, hetxl-det)')
parser.add_argument('--unc_start_value', default=0.001, type=float,
                    help='Which (average) uncertainty value to start from (higher=more variance). 0 to ignore.')
parser.add_argument('--unc_width', default=1024, type=int,
                    help='Width of the pred-net of the unc-module')
parser.add_argument('--unc_depth', default=3, type=int,
                    help="How many hidden layers the unc module should have")
group.add_argument('--pretrained', type=str2bool, default=True,
                   help='Start with pretrained version of specified network (if avail)')
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

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--iters_instead_of_epochs', type=int, default=None,
                   help="How many examples to step over before doing one validation, scheduler step, etc. Useful for big datasets where one epoch is too long. If None, use epochs as usual.")
group.add_argument('--epochs', type=int, default=113, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')

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
group.add_argument('--blur_prob', type=float, default=0.0,
                   help="With which probability to apply Gaussian Noise to the image")

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--amp', type=str2bool, default=True,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" (default: native)')
group.add_argument('--no-ddp-bb', type=str2bool, default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', type=str2bool, default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', type=str2bool, default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='./cached_datasets', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', type=str2bool, default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')

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

    # Constant for single-device training
    args.rank = 0  # global rank

    if args.is_training is None:
        args.is_training = args.split == "train"

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

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    device = torch.device(device)

    # resolve AMP arguments based on PyTorch availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
        use_amp = 'native'
        assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # set up model
    model = create_model(
        args.model,
        unc_module=args.unc_module,
        unc_width=args.unc_width,
        unc_depth=args.unc_depth,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
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

    if utils.is_primary(args):
        n_params = sum([m.numel() for m in model.parameters()])
        _logger.info(f'Model {safe_model_name(args.model)} created, param count: {n_params}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # set up automatic mixed-precision (AMP)
    amp_autocast = suppress  # do nothing
    if use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # set up the train dataset
    dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        n_few_shot=args.n_few_shot,
    )

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
        dataset = AugMixDataset(dataset, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    transform = None
    if args.ssl:
        transform = prepare_n_crop_transform([ImagenetTransform(
            interpolation=train_interpolation, crop_size=data_config['input_size'][1])],
            num_crops_per_aug=[2])
    loader = create_loader(
        dataset,
        transform=transform,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=args.is_training,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        blur_prob=args.blur_prob,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation if args.is_training else data_config["interpolation"],
        crop_pct=None if args.is_training else data_config["crop_pct"],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding
    )

    # set up cache file
    exp_name = f'hdf5_{args.dataset.split("/")[-1]}_{args.split}_{args.model}'
    output_dir = utils.get_outdir(args.output, exp_name, inc=True)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)
    output_file = os.path.join(output_dir, "dataset.hdf5")

    # Set up how many dataloader iterations to train on
    if args.iters_instead_of_epochs is None:
        batches_per_epoch = len(loader)  # Train for the full epoch
    else:
        batches_per_epoch = args.iters_instead_of_epochs // args.batch_size
    current_train_iter = iter(loader)

    time_start_epoch = datetime.now()
    _logger.info(f"Setup took {(time_start_epoch - time_start_setup).total_seconds()} seconds")
    try:
        with h5py.File(output_file, 'w') as hf:
            for epoch in range(args.epochs):
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(epoch)

                run_one_epoch(
                    epoch,
                    model,
                    loader,
                    current_train_iter,
                    batches_per_epoch,
                    args,
                    hdf5_file=hf,
                    amp_autocast=amp_autocast,
                    mixup_fn=mixup_fn
                )

                time_end_epoch = datetime.now()
                _logger.info(f"Epoch {epoch} took {(time_end_epoch - time_start_epoch).total_seconds()} seconds")
                time_start_epoch = time_end_epoch

    except KeyboardInterrupt:
        pass

    _logger.info(f"Caching finished. Output file: {output_file}")


def run_one_epoch(
        epoch,
        model,
        loader,
        current_train_iter,
        batches_per_epoch,
        args,
        hdf5_file,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        mixup_fn=None
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    model.eval()

    batch_time_m = utils.AverageMeter()
    end = time.time()

    last_idx = batches_per_epoch - 1
    for batch_idx in range(batches_per_epoch):
        try:
            (input, target) = next(current_train_iter)
        except StopIteration:
            # We've passed through the iterator. Get a new one
            current_train_iter = iter(loader)
            (input, target) = next(current_train_iter)

        last_batch = batch_idx == last_idx

        # If we get soft labels, sample a hard label from them
        if len(target.shape) == 2:
            target = (target[:,1:]).float().multinomial(1).squeeze(-1)

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            with amp_autocast():
                pred, unc, features = model(input)

        # Write features and targets to hdf5 file
        assert len(features.shape) == 2
        assert len(target.shape) == 1
        if "embed" not in hdf5_file:
            hdf5_file.create_dataset("embed", shape=(features.shape[0], features.shape[-1]),
                                     maxshape=(None, features.shape[-1]))
        else:
            hdf5_file["embed"].resize((hdf5_file["embed"].shape[0] + features.shape[0]), axis=0)
        if "label" not in hdf5_file:
            hdf5_file.create_dataset("label", shape=(target.shape[0], 1), maxshape=(None, 1))
        else:
            hdf5_file["label"].resize((hdf5_file["label"].shape[0] + target.shape[0]), axis=0)
        hdf5_file["embed"][-features.shape[0]:, :] = features.detach().cpu().numpy()
        hdf5_file["label"][-target.shape[0]:, 0] = target.detach().cpu().numpy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_time_m.update(time.time() - end)
        end = time.time()
        if last_batch or batch_idx % args.log_interval == 0:

            if utils.is_primary(args):
                _logger.info(
                    'Epoch {}: [{:>4d}/{}]  '
                    'Remaining time: {mins_left:>4d}min'.format(
                        epoch,
                        batch_idx, batches_per_epoch,
                        mins_left=round((batches_per_epoch - batch_idx) * batch_time_m.avg / 60))
                )


if __name__ == '__main__':
    main()

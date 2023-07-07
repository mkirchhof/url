#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
import numpy as np
from scipy.stats import spearmanr
from torchmetrics.functional.classification import binary_auroc as auroc

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms.functional as func_transforms

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, pct_cropped_has_bigger_uncertainty, is_pred_correct, \
    reduce_tensor, str2bool, recall_at_one, save_image
from timm.loss import CrossEntropy

try:
    from apex import amp
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
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR', default="/home/kirchhof/data/ImageNet2012",
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='soft/imagenet',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty). Can, e.g., be torch/imagenet or soft/pig')
parser.add_argument('--data-dir-downstream', metavar='DIR', default="/home/kirchhof/data",
                    help='path to root dir where downstream datasets are stored')
parser.add_argument('--dataset-downstream', nargs='+', default=["soft/cifar", "soft/treeversity1", "soft/turkey", "soft/pig", "soft/benthic"],
                    type=str, help='list dataset type + name ("<type>/<name>") for the zero-shot downstream metrics')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', type=str2bool, default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--num-heads', default=1, type=int,
                   help='Number of heads in a shallow ensemble (default: 1 -- no ensembling)')
parser.add_argument('--lambda-value', default=1.0, type=float,
                   help='Trade-off hyperparameter between task loss and uncertainty loss when using "pred-net" (default: 1.0)')
parser.add_argument('--rank_V', default=50, type=int,
                   help='Rank of V in HET-XL (default: 50)')
parser.add_argument('--unc_module', metavar='NAME', default='class-entropy',
                    help='What to use to estimate aleatoric uncertainty (none, class-entropy, embed-norm, pred-net)')
parser.add_argument('--unc_width', default=1024, type=int,
                    help='Width of the pred-net of the unc-module')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', type=str2bool, default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-interval', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', type=str2bool, default=True,
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', type=str2bool, default=False,
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', type=str2bool, default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', type=str2bool, default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', type=str2bool, default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', type=str2bool, default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', type=str2bool, default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', type=str2bool, default=False,
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, type=str2bool,
                    help='enable experimental fast-norm')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)
parser.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. LEAVE AT 0 (just here for compatibility with train.py')

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, type=str2bool,
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, type=str2bool,
                             help="Enable AOT Autograd support.")

# For distributed evaluation:
parser.add_argument('--distributed', default=False, action="store_true", help="Distributed training is NOT IMPLEMENTED. Leave to False (required for compatibility with train.py)")
parser.add_argument('--world_size', default=0, type=int, help="Distributed training is NOT IMPLEMENTED. Argument is ignored (required for compatibility with train.py)")

# For evaluating uncertainty
parser.add_argument('--crop_min', default=0.1, type=float, help="Minimal crop pct for deteriorating an image.")
parser.add_argument('--crop_max', default=0.75, type=float, help="Maximum crop pct for deteriorating an image.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='/home/kirchhof/Nextcloud/Doktorarbeit/projects/large/data/real.json', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--soft-labels', default='/home/kirchhof/Nextcloud/Doktorarbeit/projects/large/data/raters.npz', type=str, metavar='FILENAME',
                    help='raters.npz for ImageNet Real-H soft label evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, type=str2bool,
                    help='Enable batch size decay & retry for single model validation')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()

    # create model
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
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        num_heads=args.num_heads,
        rank_V=args.rank_V,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = CrossEntropy()

    # Prepare all dataloaders
    dataset_locations = {}
    dataset_locations[args.dataset] = (args.data or args.data_dir, args.workers)
    for dataset in args.dataset_downstream:
        dataset_locations[dataset] = (args.data_dir_downstream, args.workers)

    dataloaders = {}

    for name, (location, num_workers) in dataset_locations.items():
        dataset = create_dataset(
            root=location,
            name=name,
            split=args.split,
            download=args.dataset_download,
            load_bytes=args.tf_preprocessing,
            class_map=args.class_map,
            real_labels=args.real_labels,
            soft_labels=args.soft_labels,
        )

        if args.valid_labels:
            with open(args.valid_labels, 'r') as f:
                valid_labels = {int(line.rstrip()) for line in f}
                valid_labels = [i in valid_labels for i in range(args.num_classes)]
        else:
            valid_labels = None

        if args.real_labels and args.dataset == "torch/imagenet":
            real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
        else:
            real_labels = None

        crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
        dataloaders[name] = create_loader(
            dataset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=num_workers,
            crop_pct=crop_pct,
            crop_mode=data_config['crop_mode'],
            pin_memory=args.pin_mem,
            device=device,
            tf_preprocessing=args.tf_preprocessing,
        )

    # run on the upstream dataloader
    loader = dataloaders.pop(args.dataset)
    results = validate_one_epoch(model, loader, criterion, args, device, amp_autocast, valid_labels, real_labels)

    # run on the downstream dataloders (if any)
    if len(dataloaders) > 0:
        results_downstream = validate_on_downstream_datasets(model, dataloaders, criterion, args, device, amp_autocast)
        results.update(results_downstream)

    # Add some information about the model
    results["param_count"] = round(param_count / 1e6, 2)
    results["img_size"] = data_config['input_size'][-1]
    results["crop_pct"] = crop_pct
    results["interpolation"] = data_config['interpolation']

    _logger.info('Upstream * Acc@1 {:.3f} Acc@5 {:.3f} R@1 {:.3f} croppedHasBiggerUnc {:.3f} (rcorr {:.3f}) AUROC-correctness {:.3f}'.format(
       results['top1'], results['top5'], results['r1'], results['croppedHasBiggerUnc'],
       results["rcorr_crop_unc"], results["auroc_correct"]))
    if len(dataloaders) > 0:
        _logger.info('Downstream * rcorr_entropy {:.3f} croppedHasBiggerUnc {:.3f} R@1 {:.3f} AUROC-correctness {:.3f}'.format(
                results["avg_downstream_rcorr_entropy"], results["avg_downstream_croppedHasBiggerUnc"],
                results["avg_downstream_r1"], results["avg_downstream_auroc_correct"]))

    return results


def validate_on_downstream_datasets(
        model,
        loaders,
        criterion,
        args,
        device=torch.device("cuda"),
        amp_autocast=suppress,
        log_suffix=''):
    """
    This function applies validate_one_epoch to all dataloaders in the dict loaders.
    It reports the performance on each dataset plus their average.
    """
    results = {}
    for name, loader in loaders.items():
        results[name] = validate_one_epoch(model, loader, criterion, args, device, amp_autocast, log_suffix=log_suffix)

    # Summarize results
    results_downstream = {}
    for key in results[list(results.keys())[0]].keys():
        if isinstance(results[list(results.keys())[0]][key], float):
            results_downstream[key] = np.mean([loader_result[key] for name, loader_result in results.items()])
    results["avg_downstream"] = results_downstream

    # flatten output
    flattened_results = {}
    for name, dict in results.items():
        for key, value in dict.items():
            flattened_results[name + "_" + key] = value

    return flattened_results


def validate_one_epoch(
        model,
        loader,
        criterion,
        args,
        device=torch.device("cuda"),
        amp_autocast=suppress,
        valid_labels=None,
        real_labels=None,
        log_suffix=''):
    # Fix the random crops for deteriorating each image
    filepath = f'timm/data/randomcrops_{len(loader.dataset)}_{args.crop_min}_{args.crop_max}.csv'
    if os.path.exists(filepath):
        center_crop = np.loadtxt(filepath, delimiter=",")
    else:
        center_crop = np.random.random(len(loader.dataset)) * (args.crop_max - args.crop_min) + args.crop_min
        np.savetxt(filepath, center_crop, delimiter=",")

    # Track metrics
    batch_time = AverageMeter()
    #losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cropped_has_bigger_unc = AverageMeter()
    n_data = len(loader.dataset)
    bs = loader.loader.batch_size
    gt_entropies = np.zeros(n_data)
    has_gt_entropies = False
    uncertainties = torch.zeros(n_data)
    uncertainties_c = np.zeros(n_data)
    correctness = torch.zeros(n_data)
    all_features = np.zeros((n_data, model.num_features), dtype="float32")
    all_targets = np.zeros(n_data)

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    cur_idx = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Add cropped inputs
            input_c = torch.zeros_like(input)
            c_sizes = torch.zeros(input.shape[0])
            for i in range(input.shape[0]):
                # Crop each image individually because torchvision cannot do it batch-wise
                crop_size = int(round(min(input.shape[2], input.shape[3]) * center_crop[cur_idx + i]))
                c_sizes[i] = crop_size
                input_c[i] = func_transforms.resize(func_transforms.center_crop(input[i], [crop_size]),
                                                    [input.shape[2], input.shape[3]])

            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # If target is an array, it contains soft labels. Split them
            if len(target.shape) == 2:
                soft_labels = target[:, 1:]
                soft_labels = soft_labels.float() / soft_labels.sum(1).unsqueeze(1)
                soft_label_entropy = -(soft_labels * torch.maximum(soft_labels.log(), torch.ones(1, device=soft_labels.device) * -10e6)).sum(dim=-1)
                gt_entropies[cur_idx:(cur_idx + input.shape[0])] = soft_label_entropy.detach().cpu().squeeze()
                has_gt_entropies = True
                target = target[:, 0]

            """
            # Save some images from time to time
            if np.random.random() <= input.shape[0] / n_data * 10:
                rnd_idx = np.random.random_integers(0, input.shape[0] - 1)
                save_image(input[rnd_idx].cpu(), f"Class {target[rnd_idx]}, GT Entropy: {soft_label_entropy[rnd_idx]:.2f}", f"{batch_idx}_random_{soft_label_entropy[rnd_idx]:.2f}.png")
                soft_label_entropy[soft_label_entropy.isnan()] = 0
                max_entropy_idx = soft_label_entropy.detach().cpu().argmax().item()
                save_image(input[max_entropy_idx].cpu(), f"Class {target[rnd_idx]}, GT Entropy: {soft_label_entropy[max_entropy_idx]:.2f}", f"{batch_idx}_maxentr_{soft_label_entropy[max_entropy_idx]:.2f}.png")
            """

            # compute output
            with amp_autocast():
                output, unc, features = model(input)
                _, unc_c, _ = model(input_c)

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                if valid_labels is not None:
                    output = output[:, valid_labels]
                #loss = criterion(output, unc, target, features, model.get_classifier())

            if real_labels is not None:
                real_labels.add_result(output)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            if args.distributed:
                #reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                #reduced_loss = loss.data
                pass
            #losses.update(reduced_loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # Measure uncertainty metrics
            # TODO Might need to adjust this when using distributed evaluation
            cBiggerUnc = pct_cropped_has_bigger_uncertainty(unc.detach(), unc_c.detach())
            cropped_has_bigger_unc.update(cBiggerUnc.item(), input.size(0))
            uncertainties_c[cur_idx:(cur_idx + input.shape[0])] = unc_c.detach().cpu().squeeze()  # store for calculating rank correlation later
            uncertainties[cur_idx:(cur_idx + input.shape[0])] = unc.detach().cpu().squeeze()  # store for calculating AUROC later
            correctness[cur_idx:(cur_idx + input.shape[0])] = is_pred_correct(output, target).detach().cpu().squeeze()  # for AUROC later

            # Store features for R@1 computation later
            all_features[cur_idx:(cur_idx + input.shape[0]),:] = features.detach().cpu().squeeze()
            all_targets[cur_idx:(cur_idx + input.shape[0])] = target.detach().cpu().squeeze()

            cur_idx += input.shape[0]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.distributed and (batch_idx == last_idx or batch_idx % args.log_interval == 0):
                _logger.info(
                    'Test {0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f}) '
                    'CroppedHasHigherUnc: {croppedHasBiggerUnc.val:>7.3f} ({croppedHasBiggerUnc.avg:>7.3f})'.format(
                        log_suffix,
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        #loss=losses,
                        top1=top1,
                        top5=top5,
                        croppedHasBiggerUnc=cropped_has_bigger_unc
                    )
                )

    # Some sanity checks
    if np.any(np.isnan(all_features)):
        raise AssertionError("Embeddings contain NaNs.")
    if np.all(all_features == 0, 1).mean() > 0.9:
        raise AssertionError("More than 90% of embeddings are empty. Stopping because of potential I/O error.")

    # Recompute accuracy with real labels
    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg

    # Recall@1
    recall, correctness = recall_at_one(all_features, all_targets, mode="faiss")

    # Calculate rank correlation between crop amount and pred uncertainty
    rcorr_crop_unc = spearmanr(-uncertainties_c, center_crop)[0]

    # Calculate predictive uncertainty metric
    auroc_correct = auroc(-uncertainties, torch.from_numpy(correctness).int()).item()

    # Some summary statistics about uncertainties
    min_unc = uncertainties.min().item()
    avg_unc = uncertainties.mean().item()
    max_unc = uncertainties.max().item()

    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4),
        r1=round(recall, 4),
        top5=round(top5a, 4),
        croppedHasBiggerUnc=round(cropped_has_bigger_unc.avg, 4),
        rcorr_crop_unc=round(rcorr_crop_unc, 4),
        auroc_correct=round(auroc_correct, 4),
        min_unc=min_unc,
        max_unc=max_unc,
        avg_unc=avg_unc
    )

    # Calculate rank corr with gt entropy (if possible)
    if has_gt_entropies:
        is_not_nan = np.logical_not(np.isnan(gt_entropies))
        rcorr_entropy = spearmanr(uncertainties.numpy()[is_not_nan], gt_entropies[is_not_nan])[0]
        results["rcorr_entropy"]=round(rcorr_entropy, 4)

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if torch.cuda.is_available() and 'cuda' in args.device:
                torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models('convnext*', pretrained=True, exclude_filters=['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model, pretrained=True)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()



if __name__ == '__main__':
    main()

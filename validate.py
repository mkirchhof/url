#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
import numpy as np
from scipy.stats import spearmanr
from torchmetrics.functional.classification import binary_auroc as auroc

import torch
import torch.nn.parallel
import torchvision.transforms.functional as func_transforms
from timm.utils import accuracy, AverageMeter, pct_cropped_has_bigger_uncertainty, is_pred_correct, \
    reduce_tensor, recall_at_one, save_image
from timm.data.dataset import HDF5Dataset

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

def validate_on_downstream_datasets(
        model,
        loaders,
        criterion,
        args,
        device=torch.device("cuda"),
        amp_autocast=suppress,
        log_suffix='',
        tempfolder="timm/data"):
    """
    This function applies validate_one_epoch to all dataloaders in the dict loaders.
    It reports the performance on each dataset plus their average.
    """
    results = {}
    for name, loader in loaders.items():
        results[name] = validate_one_epoch(model, loader, criterion, args, device, amp_autocast,
                                           log_suffix=log_suffix + " " + name,
                                           is_upstream=False, dataset_name=name, tempfolder=tempfolder)

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
        log_suffix='Test',
        is_upstream=False,
        dataset_name='',
        tempfolder="timm/data"
    ):
    dataset_name = dataset_name.replace('/', '_')

    # Fix the random crops for deteriorating each image
    if isinstance(loader.dataset, HDF5Dataset):
        compare_with_cropped_image = False
    else:
        compare_with_cropped_image = True
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
    classification_correctness = torch.zeros(n_data)
    all_features = np.zeros((n_data, model.num_features), dtype="float32")
    all_targets = np.zeros(n_data)

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    cur_idx = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            if compare_with_cropped_image:
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
                if compare_with_cropped_image:
                    _, unc_c, _ = model(input_c)

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                if valid_labels is not None:
                    output = output[:, valid_labels]
                #loss = criterion(output, unc, target, features, model.get_classifier())

            if device.type == 'cuda':
                torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end, n=input.shape[0])
            end = time.time()

            # Save some images
            #for i in range(input.shape[0]):
            #    if target[i].cpu().item() in {16: 0, 35: 1, 36: 2, 15: 3, 29: 4, 19: 5}:
            #        save_image(input[i].cpu(), "", path=f"{tempfolder}/{dataset_name}_idx_{i + cur_idx}_class_{target[i]}.png")
            #save_image(input[unc.argmin().cpu().numpy()].cpu(), "", path=f"{tempfolder}/{unc.min().cpu().item():.3f}_{dataset_name}_example.png")
            #save_image(input[unc.argmax().cpu().numpy()].cpu(), "", path=f"{tempfolder}/{unc.max().cpu().item():.3f}_{dataset_name}_example.png")

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            class_correctness = output.argmax(dim=1) == target
            classification_correctness[cur_idx:(cur_idx + input.shape[0])] = class_correctness.detach().cpu().squeeze()
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
            if compare_with_cropped_image:
                cBiggerUnc = pct_cropped_has_bigger_uncertainty(unc.detach(), unc_c.detach())
                cropped_has_bigger_unc.update(cBiggerUnc.item(), input.size(0))
                uncertainties_c[cur_idx:(cur_idx + input.shape[0])] = unc_c.detach().cpu().squeeze()  # store for calculating rank correlation later
            uncertainties[cur_idx:(cur_idx + input.shape[0])] = unc.detach().cpu().squeeze()  # store for calculating AUROC later
            correctness[cur_idx:(cur_idx + input.shape[0])] = is_pred_correct(output, target).detach().cpu().squeeze()  # for AUROC later

            # Store features for R@1 computation later
            all_features[cur_idx:(cur_idx + input.shape[0]),:] = features.detach().cpu().squeeze()
            all_targets[cur_idx:(cur_idx + input.shape[0])] = target.detach().cpu().squeeze()

            cur_idx += input.shape[0]

            if not args.distributed and (batch_idx == last_idx or batch_idx % args.log_interval == 0):
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f}) '
                    'CroppedHasHigherUnc: {croppedHasBiggerUnc.val:>7.3f} ({croppedHasBiggerUnc.avg:>7.3f})'.format(
                        log_suffix,
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        top1=top1,
                        top5=top5,
                        croppedHasBiggerUnc=cropped_has_bigger_unc
                    )
                )

    # Some sanity checks
    if np.any(np.isnan(all_features)):
        _logger.warning("Embeddings contain NaNs. Replacing by zeros")
        all_features[np.isnan(all_features)] = 0.
    if np.all(all_features == 0, 1).mean() > 0.9:
        raise AssertionError("More than 90% of embeddings are empty. Stopping because of potential I/O error.")

    # Temporarily save results
    if is_upstream:
        # Save which indices we used
        num_indices = min(len(loader.dataset), args.max_num_samples // 2)
        if is_upstream:
            filepath_indices = f"timm/data/{num_indices}_indices_out_of_{len(loader.dataset)}.npy"
            if os.path.exists(filepath_indices):
                indices = np.load(filepath_indices)
            else:
                indices = torch.randperm(len(loader.dataset))[:num_indices].numpy()
                np.save(filepath_indices, indices)

        # Save temporary files for current run
        # These are all overridden in the next run
        np.save(f"{tempfolder}/temp_upstream_features.npy", all_features[indices])
        np.save(f"{tempfolder}/temp_upstream_targets.npy", all_targets[indices])
        torch.save(uncertainties[indices], f"{tempfolder}/temp_upstream_uncertainties.pt")
        torch.save({"upstream_name": dataset_name}, f"{tempfolder}/temp_upstream_name.pt")
    else:
        # Load temporary files
        # Program invariant: validate_one_epoch always gets called for the upstream
        # dataset before the downstream datasets. Otherwise, this algorithm will fail
        upstream_features = np.load(f"{tempfolder}/temp_upstream_features.npy")
        upstream_targets = np.load(f"{tempfolder}/temp_upstream_targets.npy")
        upstream_uncertainties = torch.load(f"{tempfolder}/temp_upstream_uncertainties.pt")
        upstream_name = torch.load(f"{tempfolder}/temp_upstream_name.pt")["upstream_name"]

        # Make both ID and OOD the same size to get a 50/50 split
        num_samples_keep = min(upstream_features.shape[0], all_features.shape[0])
        # For upstream, we can just use [:num_samples_keep] in the following, because it's already shuffled
        # For downstream, let's use random indices
        filepath_indices_downstream = f"timm/data/{num_samples_keep}_indices_out_of_{all_features.shape[0]}.npy"
        if os.path.exists(filepath_indices_downstream):
            indices_downstream = np.load(filepath_indices_downstream)
        else:
            indices_downstream = torch.randperm(all_features.shape[0])[:num_samples_keep].numpy()
            np.save(filepath_indices_downstream, indices_downstream)

        # Shift all_targets to ensure there is no overlap between ID and OOD classes
        # The exact values of the classes bear no semantic meaning
        upstream_targets_shifted = all_targets[:num_samples_keep].max() + 1 + upstream_targets

        concatenated_features = np.concatenate(
            [upstream_features[:num_samples_keep], all_features[indices_downstream]],
            axis=0,
        )
        concatenated_targets = np.concatenate(
            [
                upstream_targets_shifted[:num_samples_keep],
                all_targets[indices_downstream]
            ],
            axis=0,
        )

        # 0: ID; 1: OOD
        ood_targets = np.concatenate(
            [
                np.zeros((num_samples_keep,), dtype=np.int32),
                np.ones((num_samples_keep), dtype=np.int32)
            ],
            axis=0
        )

        concatenated_uncertainties = torch.cat(
            [
                upstream_uncertainties[:num_samples_keep],
                uncertainties[indices_downstream]
            ],
            dim=0,
        )

        saved_uncertainties = {
            "ID": upstream_uncertainties[:num_samples_keep],
            "OOD": uncertainties[indices_downstream],
        }

        torch.save(
            saved_uncertainties,
            f"{tempfolder}/uncertainties_{upstream_name}_{dataset_name}.pt"
        )

    # Metrics only on current dataset

    # Recall@1
    # Comment: Here we completely override the previously defined "correctness"
    # tensor. (L433) Is that the intended behavior or did you want to use the previous
    # one somewhere?
    # Answer: It's intended behaviour, since we switched from calculating AUROC w.r.t. Accuracy to
    # AUROC w.r.t. R@1, but I agree that we should remove the old code.


    #np.savetxt(f"{dataset_name}_features.csv", all_features, delimiter=",")
    #np.savetxt(f"{dataset_name}_targets.csv", all_targets, delimiter=",")
    #np.savetxt(f"{dataset_name}_unc.csv", uncertainties, delimiter=",")

    recall, correctness = recall_at_one(all_features, all_targets, mode="faiss")

    # Safe recall experiments
    #certain_idxes = []
    #for c_idx in np.unique(all_targets):
    #    threshold = np.quantile(uncertainties[all_targets == c_idx], 0.90)
    #    certain_idxes.extend([idx for idx in np.arange(len(all_targets))[all_targets == c_idx] if uncertainties[idx] <= threshold])
    #recall_safe_database,_ = recall_at_one(all_features, all_targets, mode="faiss", idxes_database=certain_idxes)
    #recall_safe_database_and_queries,_ = recall_at_one(all_features, all_targets, mode="faiss", idxes_database=certain_idxes, idxes_query=certain_idxes)
    #recall_safe_queries, _ = recall_at_one(all_features, all_targets, mode="faiss", idxes_query=certain_idxes)
    #print(recall, recall_safe_database, recall_safe_database_and_queries, recall_safe_queries)

    # Calculate rank correlation between crop amount and pred uncertainty
    rcorr_crop_unc = spearmanr(-uncertainties_c, center_crop)[0] if compare_with_cropped_image else None

    # Calculate predictive uncertainty metric
    auroc_correct = auroc(-uncertainties, torch.from_numpy(correctness).int()).item()

    # Calculate predictive uncertainty w.r.t. classification (not R@1)
    auroc_classification = auroc(-uncertainties, classification_correctness.int()).item()

    # Some summary statistics about uncertainties
    min_unc = uncertainties.min().item()
    avg_unc = uncertainties.mean().item()
    max_unc = uncertainties.max().item()
    unc_q10 = np.quantile(uncertainties, 0.1).item()
    unc_q25 = np.quantile(uncertainties, 0.25).item()
    unc_q50 = np.quantile(uncertainties, 0.50).item()
    unc_q75 = np.quantile(uncertainties, 0.75).item()
    unc_q90 = np.quantile(uncertainties, 0.9).item()

    results = OrderedDict(
        model=args.model,
        top1=round(top1.avg, 4),
        r1=round(recall, 4),
        top5=round(top5.avg, 4),
        croppedHasBiggerUnc=round(cropped_has_bigger_unc.avg, 4),
        rcorr_crop_unc=round(rcorr_crop_unc, 4) if compare_with_cropped_image else None,
        auroc_correct=round(auroc_correct, 4),
        auroc_classification=round(auroc_classification, 4),
        min_unc=min_unc,
        max_unc=max_unc,
        avg_unc=avg_unc,
        time_per_sample=batch_time.avg,
        unc_q10=unc_q10,
        unc_q25=unc_q25,
        unc_q50=unc_q50,
        unc_q75=unc_q75,
        unc_q90=unc_q90,
        unc_iqr_90_10=unc_q90 - unc_q10,
        unc_iqr_75_25=unc_q75 - unc_q25
    )

    # Calculate rank corr with gt entropy (if possible)
    if has_gt_entropies:
        is_not_nan = np.logical_not(np.isnan(gt_entropies))
        rcorr_entropy = spearmanr(uncertainties.numpy()[is_not_nan], gt_entropies[is_not_nan])[0]
        results["rcorr_entropy"]=round(rcorr_entropy, 4)
    
    if not is_upstream:
        # Metrics on mixed upstream/downstream dataset

        # Recall@1
        _, correctness_mixed = recall_at_one(
            concatenated_features, concatenated_targets, mode="faiss"
        )

        # Calculate predictive uncertainty metric
        auroc_correct_mixed = auroc(
            -concatenated_uncertainties,
            torch.from_numpy(correctness_mixed).int()
        ).item()

        # Calculate epistemic uncertainty metric
        # Note: we use the same uncertainties, only the benchmark is different
        auroc_ood = auroc(
            concatenated_uncertainties,
            torch.from_numpy(ood_targets)
        ).item()

        results.update(
            {
                "auroc_correct_mixed": auroc_correct_mixed,
                "auroc_ood": auroc_ood
            }
        )

    return results

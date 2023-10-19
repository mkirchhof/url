""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
"""
import csv
import os
from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(
        epoch,
        train_metrics,
        eval_metrics,
        filename,
        best_eval_metrics=None,
        test_metrics=None,
        best_test_metrics=None,
        test_metrics2=None,
        best_test_metrics2=None,
        lr=None,
        write_header=False,
        log_wandb=False,
):
    rowd = OrderedDict(epoch=epoch)
    if train_metrics is not None:
        rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    if eval_metrics is not None:
        rowd.update([('eval_' + k, v) for k, v in eval_metrics.items()])
    if best_eval_metrics is not None:
        rowd.update([('best_eval_' + k, v) for k, v in best_eval_metrics.items()])
    if test_metrics is not None:
        rowd.update([('test_' + k, v) for k, v in test_metrics.items()])
    if best_test_metrics is not None:
        rowd.update([('best_test_' + k, v) for k, v in best_test_metrics.items()])
    if test_metrics2 is not None:
        rowd.update([('furthertest_' + k, v) for k, v in test_metrics2.items()])
    if best_test_metrics2 is not None:
        rowd.update([('best_furthertest_' + k, v) for k, v in best_test_metrics2.items()])
    if lr is not None:
        rowd['lr'] = lr
    if log_wandb:
        wandb.log(rowd)
    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

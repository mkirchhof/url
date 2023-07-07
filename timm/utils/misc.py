""" Misc utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import argparse
import ast
import re
import numpy as np
from numpy import i0  # modified Bessel function of first kind order 0, I_0
from scipy.special import ive  # exponential modified Bessel function of first kind, I_v * exp(-abs(kappa))
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def add_bool_arg(parser, name, default=False, help=''):
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, type=str2bool, default=False, help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', default=True, help=help)
    parser.set_defaults(**{dest_name: default})


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def str2bool(v):
    """
    Thank to stackoverflow user: Maxim
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    :param v: A command line argument with values [yes, true, t, y, 1, True, no, false, f, n, 0, False]
    :return: Boolean version of the command line argument
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def log_vmf_norm_const(kappa, dim=10):
    # Approximates the log vMF normalization constant (for the ELK loss)
    # See approx_vmf_norm_const.R to see how it was approximated

    if dim==4:
        return -0.826604 - 0.354357 * kappa - 0.383723 * kappa**1.1
    if dim==8:
        return -1.29737 + 0.36841 * kappa - 0.80936 * kappa**1.1
    elif dim==10:
        return -1.27184 + 0.67365 * kappa - 0.98726 * kappa**1.1
    elif dim==16:
        return -0.23773 + 1.39146 * kappa - 1.39819 * kappa**1.1
    elif dim==32:
        return 8.07579 + 2.28954 * kappa - 1.86925 * kappa**1.1
    elif dim==64:
        return 38.82967 + 2.34269 * kappa - 1.77425 * kappa**1.1
    elif dim==512:
        return 866.3 + 0.1574 * kappa - 0.0236 * kappa**1.5
    elif dim>=1024:
        return 2095 - 0.01204 * kappa - 0.0007967 * kappa**1.9
        # For higher dimensions the numerical Bessel approximations fail,
        # so the best we can do is use the estimate of dim 1024
    else:
        return np.log(_vmf_normalize(kappa, dim))


def _vmf_normalize(kappa, dim):
    """Compute normalization constant using built-in numpy/scipy Bessel
    approximations.
    Works well on small kappa and mu.
    Imported from https://github.com/jasonlaska/spherecluster/blob/develop/spherecluster/von_mises_fisher_mixture.py
    """
    if kappa < 1e-15:
        kappa = 1e-15

    num = (dim / 2.0 - 1.0) * np.log(kappa)

    if dim / 2.0 - 1.0 < 1e-15:
        denom = (dim / 2.0) * np.log(2.0 * np.pi) + np.log(i0(kappa))
    else:
        denom = (dim / 2.0) * np.log(2.0 * np.pi) + np.log(ive(dim / 2.0 - 1.0, kappa)) + kappa

    if np.isinf(num):
        raise ValueError("VMF scaling numerator was inf.")

    if np.isinf(denom):
        raise ValueError("VMF scaling denominator was inf.")

    const = np.exp(num - denom)

    if const == 0:
        raise ValueError("VMF norm const was 0.")

    return const


def plt_image(im):
    plt.imshow(
        torch.minimum(torch.ones(1), torch.maximum(torch.zeros(1),
        im.cpu().permute(1, 2, 0) * torch.tensor(IMAGENET_DEFAULT_STD) + torch.tensor(IMAGENET_DEFAULT_MEAN)))
    )
    plt.show()

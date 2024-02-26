""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""
import os
import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, KMNIST, FashionMNIST, ImageFolder
try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False
try:
    from torchvision.datasets import QMNIST
    has_qmnist = True
except ImportError:
    has_qmnist = False
try:
    from torchvision.datasets import ImageNet
    from .readers.reader_imagenetrealh import SoftImageNet
    has_imagenet = True
except ImportError:
    has_imagenet = False
try:
    from .readers.reader_imagenet8k import ImageNet8k
    has_imagenet8k = True
except ImportError:
    has_imagenet8k = False

from .dataset import IterableImageDataset, ImageDataset, HDF5Dataset

_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training=False,
        download=False,
        batch_size=None,
        seed=42,
        repeats=0,
        model_name="",
        **kwargs
):
    """ Dataset factory method

    In parenthesis after each arg are the type of dataset supported for each arg, one of:
      * folder - default, timm folder (or tar) based ImageDataset
      * torch - torchvision based datasets
      * HFDS - Hugging Face Datasets
      * TFDS - Tensorflow-datasets wrapper in IterableDataset interface via IterableImageDataset
      * WDS - Webdataset
      * all - any of the above

    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
        search_split: search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (HFDS, TFDS, torch)
        is_training: create dataset in train mode, this is different from the split.
            For Iterable / TFDS it enables shuffle, ignored for other datasets. (TFDS, WDS)
        batch_size: batch size hint for (TFDS, WDS)
        seed: seed for iterable datasets (TFDS, WDS)
        repeats: dataset repeats per iteration i.e. epoch (TFDS, WDS)
        **kwargs: other args to pass to dataset

    Returns:
        Dataset object
    """
    name = name.lower()
    if name.startswith('torch/'):
        name = name.split('/', 2)[-1]
        torch_kwargs = dict(root=root, **kwargs)
        if name in _TORCH_BASIC_DS:
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            ds = ds_class(train=use_train, **torch_kwargs)
        elif name == 'inaturalist' or name == 'inat':
            assert has_inaturalist, 'Please update to PyTorch 1.10, torchvision 0.11+ for Inaturalist'
            target_type = 'full'
            split_split = split.split('/')
            if len(split_split) > 1:
                target_type = split_split[0].split('_')
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = '2021_train'
            elif split in _EVAL_SYNONYM:
                split = '2021_valid'
            ds = INaturalist(version=split, target_type=target_type, **torch_kwargs)
        elif name == 'places365':
            assert has_places365, 'Please update to a newer PyTorch and torchvision for Places365 dataset.'
            if split in _TRAIN_SYNONYM:
                split = 'train-standard'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            ds = Places365(split=split, **torch_kwargs)
        elif name == 'qmnist':
            assert has_qmnist, 'Please update to a newer PyTorch and torchvision for QMNIST dataset.'
            use_train = split in _TRAIN_SYNONYM
            ds = QMNIST(train=use_train, **torch_kwargs)
        elif name == 'imagenet':
            assert has_imagenet, 'Please update to a newer PyTorch and torchvision for ImageNet dataset.'
            if split in _EVAL_SYNONYM:
                split = 'val'
            if "n_few_shot" in torch_kwargs:
                torch_kwargs.pop("n_few_shot")
            ds = ImageNet(split=split, **torch_kwargs)
        elif name.startswith('imagenet'):
            assert has_imagenet, 'Please update to a newer PyTorch and torchvision for ImageNet dataset.'
            if split in _EVAL_SYNONYM:
                split = 'val'
            if "n_few_shot" in torch_kwargs:
                torch_kwargs.pop("n_few_shot")
            ds = ImageNet(split=split, **torch_kwargs)
            if name.endswith("_certain_50_pct") or name.endswith("_certain_80_pct") or name.endswith("_certain_90_pct")\
                    or name.endswith("_certain_95_pct"):
                if name.endswith("_certain_50_pct"):
                    col = 2
                elif name.endswith("_certain_80_pct"):
                    col = 3
                elif name.endswith("_certain_90_pct"):
                    col = 4
                elif name.endswith("_certain_95_pct"):
                    col = 5
                indices = np.loadtxt("data/uncertain_indices_imagenet1k_train.csv", delimiter=",", skiprows=1)[:,col]
                ds.imgs = [x for x, include in zip(ds.imgs, indices) if include]
                ds.samples = [x for x, include in zip(ds.samples, indices) if include]
                ds.targets = [x for x, include in zip(ds.targets, indices) if include]
        elif name == 'image_folder' or name == 'folder':
            # in case torchvision ImageFolder is preferred over timm ImageDataset for some reason
            if search_split and os.path.isdir(root):
                # look for split specific sub-folder in root
                root = _search_split(root, split)
            ds = ImageFolder(root, **kwargs)
        else:
            assert False, f"Unknown torchvision dataset {name}"
    elif name.startswith('hfds/'):
        # NOTE right now, HF datasets default arrow format is a random-access Dataset,
        # There will be a IterableDataset variant too, TBD
        ds = ImageDataset(root, reader=name, split=split, class_map=class_map, **kwargs)
    elif name.startswith('tfds/') or name.startswith('vtab/'):
        if split in _EVAL_SYNONYM:
            split = 'val'
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            is_training=is_training,
            download=download,
            batch_size=batch_size,
            repeats=repeats,
            seed=seed,
            **kwargs
        )
    elif name.startswith('wds/'):
        ds = IterableImageDataset(
            root,
            reader=name,
            split=split,
            is_training=is_training,
            batch_size=batch_size,
            repeats=repeats,
            seed=seed,
            **kwargs
        )
    elif name.startswith('repr/') or name.startswith('soft/'):
        if name.split('/', 2)[-1] == "imagenet":
            assert has_imagenet, 'Please update to a newer PyTorch and torchvision for ImageNet dataset.'
            assert split in _EVAL_SYNONYM, "soft/imagenet is only available for the validation dataset"
            ds = SoftImageNet(root, path_soft_labels="/home/kirchhof/Nextcloud/Doktorarbeit/projects/large/data/raters.npz", path_real_labels="/home/kirchhof/Nextcloud/Doktorarbeit/projects/large/data/real.json", **kwargs)
        else:
            ds = IterableImageDataset(
                root,
                reader=name,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                repeats=repeats,
                seed=seed,
                **kwargs
            )
    elif name.startswith('folder/imagenet8k'):
        assert has_imagenet8k
        if split in _EVAL_SYNONYM:
            split = 'val'
        torch_kwargs = dict(root=root, **kwargs)
        if "n_few_shot"in torch_kwargs:
            torch_kwargs.pop("n_few_shot")
        ds = ImageNet8k(split=split, reduce_size=name.endswith("_small"), **torch_kwargs)
    elif name.startswith('hdf5'):
        ds = HDF5Dataset(
            root,
            f"{name.split('/', 2)[-1]}_{split}_{model_name}"
        )
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        if search_split and os.path.isdir(root):
            # look for split specific sub-folder in root
            root = _search_split(root, split)
        ds = ImageDataset(root, reader=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
    return ds

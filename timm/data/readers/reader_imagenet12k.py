# This implements the 12k subset of ImageNet 2011 Fall that Ross Wrightman used for some pretrained models.
# We only use the same classes and val split, not necessarily the same augmentations.
#
# Besides ImageNet-21k Fall 2011 (not Winter 2021!), you'll need
# imagenet_12k_labels.txt from https://github.com/rwightman/imagenet-12k/blob/main/tfds/imagenet_12k_labels.txt and
# val_12k.csv from https://huggingface.co/datasets/rwightman/imagenet-12k-metadata/tree/main
# and place them in the ./data folder

from .reader_image_in_tar import ReaderImageInTar
from .class_map import load_class_map
import pandas as pd

class ImageNet12k(ReaderImageInTar):
    def __init__(self, root, is_val=False, cached_tarfiles=True, cache_tarinfo=True):
        # Preprocess dictionary
        class_map = load_class_map("./data/imagenet_12k_labels.txt")

        super().__init__(root,
                         class_map,
                         cache_tarfiles=cached_tarfiles,
                         cache_tarinfo=cache_tarinfo)

        # Restrict to train/val
        val_filenames = pd.read_csv("./data/val_12k.csv")["filename"].to_list()
        val_filenames = set([filename.split("/")[1] for filename in val_filenames])  # set() to make it faster
        keep_idx = [i for i, obj in enumerate(self.samples[:,0]) if (obj.name in val_filenames) == is_val]
        self.samples = self.samples[keep_idx]
        self.targets = self.targets[keep_idx]

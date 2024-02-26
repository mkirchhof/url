# This implements the ImageNet 21k Winter 2021 dataset.
#
# Please download the class names from here https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt
# and place them under ./data/imagenet21k_wordnet_ids.txt

from .reader_image_in_tar import ReaderImageInTar
from .class_map import load_class_map
import pandas as pd

class ImageNet21k(ReaderImageInTar):
    def __init__(self, root, cached_tarfiles=True, cache_tarinfo=True):
        # Preprocess dictionary
        class_map = load_class_map("./data/imagenet21k_wordnet_ids.txt")

        super().__init__(root,
                         class_map,
                         cache_tarfiles=cached_tarfiles,
                         cache_tarinfo=cache_tarinfo)

        # Note: There is no official train/val split for ImageNet-21k.

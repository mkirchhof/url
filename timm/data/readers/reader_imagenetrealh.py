# This file contains utils and a dataloader to load the downstream datasets
# of the "Is One Annotation Enough?" paper
from typing import Any, Tuple

import numpy as np
import json
import os
from torchvision.datasets import ImageNet


class SoftImageNet(ImageNet):
    def __init__(self, root: str, path_soft_labels: str, path_real_labels: str, **kwargs: Any) -> None:
        super().__init__(root, split="val", **kwargs)
        self.soft_labels, self.filepath_to_softid = load_raw_annotations(path_soft_labels, path_real_labels)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, label = super().__getitem__(index)
        soft_label = self.soft_labels[self.filepath_to_softid[os.path.split(self.samples[index][0])[-1]],:]
        all_label = np.concatenate((np.array([label]), soft_label), 0)
        return img, all_label


def load_raw_annotations(path_soft_labels, path_real_labels):
    # Loads the raw annotations from raters.npz from github.com/google-research/reassessed-imagenet
    # Adapted from github.com/google/uncertainty-baselines/blob/main/baselines/jft/data_uncertainty_utils.py#L87
    data = np.load(path_soft_labels)

    summed_ratings = np.sum(data['tensor'], axis=0) # 0 is the annotator axis
    yes_prob = summed_ratings[:, 2]
    # This gives a [questions] np array.
    # It gives how often the question "Is image X of class Y" was answered with "yes".

    # We now need to summarize these questions across the images and labels
    num_labels = 1000
    soft_labels = {}
    for idx, (file_name, label_id) in enumerate(data['info']):
        if file_name not in soft_labels:
            soft_labels[file_name] = np.zeros(num_labels, dtype=np.int64)
        added_label = np.zeros(num_labels, dtype=np.int64)
        added_label[int(label_id)] = yes_prob[idx]
        soft_labels[file_name] = soft_labels[file_name] + added_label

    # Questions were only asked about 24889 images, and of those 1067 have no single yes vote at any label
    # We will fill up (some of) the missing ones by taking the ImageNet Real Labels
    new_soft_labels = {}
    with open(path_real_labels) as f:
        real_labels = json.load(f)
    for idx, label in enumerate(real_labels):
        key = 'ILSVRC2012_val_'
        key += (8 - len(str(idx + 1))) * '0' + str(idx + 1) + '.JPEG'
        if len(label) > 0:
            one_hot_label = np.zeros(num_labels, dtype=np.int64)
            one_hot_label[label] = 1
            new_soft_labels[key] = one_hot_label
        else:
            new_soft_labels[key] = np.zeros(num_labels)

    # merge soft and hard labels
    unique_img_filepath = list(new_soft_labels.keys())
    filepath_to_imgid = dict(zip(unique_img_filepath, list(np.arange(0, len(unique_img_filepath)))))
    soft_labels_array = np.zeros((len(unique_img_filepath), 1000), dtype=np.int64)
    for idx, img in enumerate(unique_img_filepath):
        if img in soft_labels and soft_labels[img].sum() > 0:
            final_soft_label = soft_labels[img]
        else:
            final_soft_label = new_soft_labels[img]
        soft_labels_array[idx,:] = final_soft_label

    # Note that 750 of the 50000 images in soft_labels_array will still not have a label at all.
    # These are ones where the old imagenet label was false and also the raters could not determine any new one.
    # We hand 0 matrices out for them. They should be ignored in computing the metrics

    return soft_labels_array, filepath_to_imgid

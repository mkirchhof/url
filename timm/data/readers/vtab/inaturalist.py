# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements INaturalist data class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .base import ImageTfdsData
import tensorflow_datasets as tfds


TRAIN_SPLIT_PERCENT = 90


class INaturalistData(ImageTfdsData):
  """INaturalist dataset."""

  def __init__(self, year=2017, data_dir=None):
    supported_years = [2017]
    if year not in supported_years:
      raise ValueError(
          "Only competitions from years {!r} are supported, but {!r} was given"
          .format(supported_years, year))
    dataset_builder = tfds.builder(
        "i_naturalist{}:0.1.0".format(year), data_dir=data_dir)
    dataset_builder.download_and_prepare()

    tfds_splits = {
        "train": "train[:{}%]".format(TRAIN_SPLIT_PERCENT),
        "val": "train[{}%:]".format(TRAIN_SPLIT_PERCENT),
        "trainval": "train",
        "test": "validation"
    }

    # Example counts are retrieved from the tensorflow dataset info.
    trainval_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
    train_count = int(round(trainval_count * TRAIN_SPLIT_PERCENT / 100.0))
    val_count = trainval_count - train_count
    test_count = dataset_builder.info.splits[tfds.Split.VALIDATION].num_examples

    # Creates a dict with example counts for each split.
    num_samples_splits = {
        "train": train_count,
        "val": val_count,
        "trainval": trainval_count,
        "test": test_count
    }

    super(INaturalistData, self).__init__(
        dataset_builder=dataset_builder,
        tfds_splits=tfds_splits,
        num_samples_splits=num_samples_splits,
        num_classes=dataset_builder.info.features["label"].num_classes)

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

"""Abstract class for reading the data using tfds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


def make_get_tensors_fn(output_tensors):
  """Create a function that outputs a collection of tensors from the dataset."""

  def _get_fn(data):
    """Get tensors by name."""
    return {tensor_name: data[tensor_name] for tensor_name in output_tensors}

  return _get_fn


def make_get_and_cast_tensors_fn(output_tensors):
  """Create a function that gets and casts a set of tensors from the dataset.

  Optionally, you can also rename the tensors.

  Examples:
    # This simply gets "image" and "label" tensors without any casting.
    # Note that this is equivalent to make_get_tensors_fn(["image", "label"]).
    make_get_and_cast_tensors_fn({
      "image": None,
      "label": None,
    })

    # This gets the "image" tensor without any type conversion, casts the
    # "heatmap" tensor to tf.float32, and renames the tensor "class/label" to
    # "label" and casts it to tf.int64.
    make_get_and_cast_tensors_fn({
      "image": None,
      "heatmap": tf.float32,
      "class/label": ("label", tf.int64),
    })

  Args:
    output_tensors: dictionary specifying the set of tensors to get and cast
      from the dataset.

  Returns:
    The function performing the operation.
  """

  def _tensors_to_cast():
    tensors_to_cast = []  # AutoGraph does not support generators.
    for tensor_name, tensor_dtype in output_tensors.items():
      if isinstance(tensor_dtype, tuple) and len(tensor_dtype) == 2:
        tensors_to_cast.append((tensor_name, tensor_dtype[0], tensor_dtype[1]))
      elif tensor_dtype is None or isinstance(tensor_dtype, tf.dtypes.DType):
        tensors_to_cast.append((tensor_name, tensor_name, tensor_dtype))
      else:
        raise ValueError('Values of the output_tensors dictionary must be '
                         'None, tf.dtypes.DType or 2-tuples.')
    return tensors_to_cast

  def _get_and_cast_fn(data):
    """Get and cast tensors by name, optionally changing the name too."""

    return {
        new_name:
        data[name] if new_dtype is None else tf.cast(data[name], new_dtype)
        for name, new_name, new_dtype in _tensors_to_cast()
    }

  return _get_and_cast_fn


def compose_preprocess_fn(*functions):
  """Compose two or more preprocessing functions.

  Args:
    *functions: Sequence of preprocess functions to compose.

  Returns:
    The composed function.
  """

  def _composed_fn(x):
    for fn in functions:
      if fn is not None:  # Note: If one function is None, equiv. to identity.
        x = fn(x)
    return x

  return _composed_fn


# Note: DO NOT implement any method in this abstract class.
@six.add_metaclass(abc.ABCMeta)
class ImageDataInterface(object):
  """Interface to the image data classes."""

  @property
  @abc.abstractmethod
  def default_label_key(self):
    """Returns the default label key of the dataset."""

  @property
  @abc.abstractmethod
  def label_keys(self):
    """Returns a tuple with the available label keys of the dataset."""

  @property
  @abc.abstractmethod
  def num_channels(self):
    """Returns the number of channels of the images in the dataset."""

  @property
  @abc.abstractmethod
  def splits(self):
    """Returns the splits defined in the dataset."""

  @abc.abstractmethod
  def get_num_samples(self, split_name):
    """Returns the number of images in the given split name."""

  @abc.abstractmethod
  def get_num_classes(self, label_key=None):
    """Returns the number of classes of the given label_key."""


class ImageData(ImageDataInterface):
  """Abstract data provider class.

  IMPORTANT: You should use ImageTfdsData below whenever is posible. We want
  to use as many datasets in TFDS as possible to ensure reproducibility of our
  experiments. Your data class should only inherit directly from this if you
  are doing experiments while creating a TFDS dataset.
  """

  @abc.abstractmethod
  def __init__(self,
               num_samples_splits,
               num_classes,
               default_label_key='label',
               image_decoder=None,
               composed_decoder=None,
               postprocess_fn=None,
               num_channels=3):
    """Initializer for the base ImageData class.

    Args:
      num_samples_splits: a dictionary, that maps splits ("train", "trainval",
          "val", and "test") to the corresponding number of samples.
      num_classes: int/dict, number of classes in this dataset for the
        `default_label_key` tensor, or dictionary with the number of classes in
        each label tensor.
      default_label_key: optional, string with the name of the tensor to use
        as label. Default is "label".
      image_decoder: a function to decode image.
      num_channels: number of channels in the dataset image.
    """
    self._log_warning_if_direct_inheritance()
    self._num_samples_splits = num_samples_splits
    self._default_label_key = default_label_key
    self._image_decoder = image_decoder
    self._num_channels = num_channels
    self.composed_decoder = composed_decoder
    self.postprocess_fn = postprocess_fn

    if isinstance(num_classes, dict):
      self._num_classes = num_classes
      if default_label_key not in num_classes:
        raise ValueError(
            'No num_classes was specified for the default_label_key %r' %
            default_label_key)
    elif isinstance(num_classes, int):
      self._num_classes = {default_label_key: num_classes}
    else:
      raise ValueError(
          '"num_classes" must be a int or a dict, but type %r was given' %
          type(num_classes))

  @property
  def default_label_key(self):
    return self._default_label_key

  @property
  def label_keys(self):
    return tuple(self._num_classes.keys())

  @property
  def num_channels(self):
    return self._num_channels

  @property
  def splits(self):
    return tuple(self._num_samples_splits.keys())

  def get_num_samples(self, split_name):
    return self._num_samples_splits[split_name]

  def get_num_classes(self, label_key=None):
    if label_key is None:
      label_key = self._default_label_key
    return self._num_classes[label_key]

  def get_version(self):
    return NotImplementedError('Version is not supported outside TFDS.')

  @abc.abstractmethod
  def _get_dataset_split(self, split_name, shuffle_files=False):
    """Return the Dataset object for the given split name.

    Args:
      split_name: Name of the dataset split to get.
      shuffle_files: Whether or not to shuffle files in the dataset.

    Returns:
      A tf.data.Dataset object containing the data for the given split.
    """

  def _log_warning_if_direct_inheritance(self):
    tf.logging.warning(
        'You are directly inheriting from ImageData. Please, consider porting '
        'your dataset to TFDS (go/tfds) and inheriting from ImageTfdsData '
        'instead.')


class ImageTfdsData(ImageData):
  """Abstract data provider class for datasets available in Tensorflow Datasets.

  To add new datasets inherit from this class. This class implements a simple
  API that is used throughout the project and provides standardized way of data
  preprocessing and batching.
  """

  @abc.abstractmethod
  def __init__(self, dataset_builder, tfds_splits, image_key='image', **kwargs):
    """Initializer for the base ImageData class.

    Args:
      dataset_builder: tfds dataset builder object.
      tfds_splits: a dictionary, that maps splits ("train", "trainval", "val",
          and "test") to the corresponding tfds `Split` objects.
      image_key: image key.
      **kwargs: Additional keyword arguments for the ImageData class.
    """
    self._dataset_builder = dataset_builder
    self._tfds_splits = tfds_splits
    self._image_key = image_key

    # Overwrite image decoder
    def _image_decoder(data):
      decoder = dataset_builder.info.features[image_key].decode_example
      data[image_key] = decoder(data[image_key])
      return data
    self._image_decoder = _image_decoder

    kwargs.update({'image_decoder': _image_decoder})

    super(ImageTfdsData, self).__init__(**kwargs)

  def get_version(self):
    return self._dataset_builder.version.__str__()

  def _get_dataset_split(self, split_name, shuffle_files):
    dummy_decoder = tfds.decode.SkipDecoding()
    return self._dataset_builder.as_dataset(
        split=self._tfds_splits[split_name], shuffle_files=shuffle_files,
        decoders={self._image_key: dummy_decoder})

  def _log_warning_if_direct_inheritance(self):
    pass

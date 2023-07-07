# This file contains utils and a dataloader to load the downstream datasets
# of the "Is One Annotation Enough?" paper

import numpy as np
import os
from torchvision.datasets.folder import pil_loader
import pandas as pd

from .reader import Reader
from .img_extensions import get_img_extensions


class ReaderCUB(Reader):

    def __init__(self, root, split="test", **kwargs):
        """
        root: string, path to root directory of dataset
        split: string, train/validation/test/all. Which folds to use
        """
        super().__init__()

        ### Prepare files
        self.root = root
        image_sourcepath = root + '/images'
        image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x],
                               key=lambda x: int(x.split('.')[0]))
        image_list = {int(key.split('.')[0]) - 1: sorted(
            [image_sourcepath + '/' + key + '/' + x for x in os.listdir(image_sourcepath + '/' + key) if '._' not in x])
                      for key in image_classes}
        image_list = [[(key, img_path) for img_path in image_list[key]] for key in image_list.keys()]
        image_list = [x for y in image_list for x in y]

        ### Dictionary of structure key=class, value=list_of_samples_with_said_class
        image_dict = {}
        for key, img_path in image_list:
            if not key in image_dict.keys():
                image_dict[key] = []
            image_dict[key].append(img_path)

        ### Use the first half of the sorted data as training and the second half as test set
        keys = sorted(list(image_dict.keys()))
        train, test = keys[:len(keys) // 2], keys[len(keys) // 2:]
        test_image_dict = {key: image_dict[key] for key in test}

        ### Split the training data into a train/val setup by class.
        if split != "test":
            train_image_dict, val_image_dict = {}, {}
            for key in train:
                train_ixs = np.array(list(set(np.round(
                    np.linspace(0, len(image_dict[key]) - 1, int(len(image_dict[key]) * 0.5)))))).astype(int)
                val_ixs = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key] = np.array(image_dict[key])[val_ixs]

        ### choose the correct split
        if split == "train":
            self.samples = train_image_dict
        elif split == "validation":
            self.samples = val_image_dict
        elif split == "test" or split == "trainontest" or split == "testontest":
            self.samples = test_image_dict
        # Now flatten all classes into one dict
        self.samples = [[(x, key) for x in self.samples[key]] for key in self.samples.keys()]
        self.samples = [x for y in self.samples for x in y]
        if split == "trainontest":
            min_class_idx = np.min([i[-1] for i in self.samples]).item()
            self.samples = [(self.samples[i][0], self.samples[i][-1] - min_class_idx) for i in np.arange(0, len(self.samples), 2)]
        elif split == "testontest":
            min_class_idx = np.min([i[-1] for i in self.samples]).item()
            self.samples = [(self.samples[i][0], self.samples[i][-1] - min_class_idx) for i in np.arange(1, len(self.samples), 2)]

        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. '
                f'Supported image extensions are {", ".join(get_img_extensions())}')

    def __getitem__(self, index):
        # Return format is
        # 1) PIL image
        # 2) array where first column is target label and remaining columns are raw label counts
        path = self.samples[index][0]
        label = self.samples[index][-1]
        path = os.path.join(self.root, path)
        return pil_loader(path), np.array(label)

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


class ReaderCARS(Reader):

    def __init__(self, root, split="test", **kwargs):
        """
        root: string, path to root directory of dataset
        split: string, train/validation/test/all. Which folds to use
        """
        super().__init__()

        ### Prepare files
        self.root = root
        image_sourcepath = root + '/images'
        image_classes = sorted([x for x in os.listdir(image_sourcepath)])
        image_list = {
            i: sorted([image_sourcepath + '/' + key + '/' + x for x in os.listdir(image_sourcepath + '/' + key)]) for
            i, key in enumerate(image_classes)}
        image_list = [[(key, img_path) for img_path in image_list[key]] for key in image_list.keys()]
        image_list = [x for y in image_list for x in y]

        ### Dictionary of structure key=class, value=list_of_samples_with_said_class
        image_dict = {}
        for key, img_path in image_list:
            if not key in image_dict.keys():
                image_dict[key] = []
            image_dict[key].append(img_path)

        ### Use the first half of the sorted data as training and the second half as test set
        keys = sorted(list(image_dict.keys()))
        train, test = keys[:len(keys) // 2], keys[len(keys) // 2:]
        test_image_dict = {key: image_dict[key] for key in test}

        ### Split the training data into a train/val setup by class.
        if split != "test":
            train_image_dict, val_image_dict = {}, {}
            for key in train:
                train_ixs = np.array(list(set(np.round(
                    np.linspace(0, len(image_dict[key]) - 1, int(len(image_dict[key]) * 0.5)))))).astype(int)
                val_ixs = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key] = np.array(image_dict[key])[val_ixs]

        ### choose the correct split
        if split == "train":
            self.samples = train_image_dict
        elif split == "validation":
            self.samples = val_image_dict
        elif split == "test" or split == "trainontest" or split == "testontest":
            self.samples = test_image_dict
        # Now flatten all classes into one dict
        self.samples = [[(x, key) for x in self.samples[key]] for key in self.samples.keys()]
        self.samples = [x for y in self.samples for x in y]
        if split == "trainontest":
            min_class_idx = np.min([i[-1] for i in self.samples]).item()
            self.samples = [(self.samples[i][0], self.samples[i][-1] - min_class_idx) for i in np.arange(0, len(self.samples), 2)]
        elif split == "testontest":
            min_class_idx = np.min([i[-1] for i in self.samples]).item()
            self.samples = [(self.samples[i][0], self.samples[i][-1] - min_class_idx) for i in np.arange(1, len(self.samples), 2)]

        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. '
                f'Supported image extensions are {", ".join(get_img_extensions())}')

    def __getitem__(self, index):
        # Return format is
        # 1) PIL image
        # 2) array where first column is target label and remaining columns are raw label counts
        path = self.samples[index][0]
        label = self.samples[index][-1]
        path = os.path.join(self.root, path)
        return pil_loader(path), np.array(label)

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename


class ReaderSOP(Reader):

    def __init__(self, root, split="test", **kwargs):
        """
        root: string, path to root directory of dataset
        split: string, train/validation/test/all. Which folds to use
        """
        super().__init__()

        ### Prepare files
        self.root = root
        image_sourcepath = root + '/images'
        training_files = pd.read_table(root + '/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
        test_files = pd.read_table(root + '/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

        spi = np.array([(a, b) for a, b in zip(training_files['super_class_id'], training_files['class_id'])])
        super_dict = {}
        super_conversion = {}
        for i, (super_ix, class_ix, image_path) in enumerate(
                zip(training_files['super_class_id'], training_files['class_id'], training_files['path'])):
            if super_ix not in super_dict: super_dict[super_ix] = {}
            if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
            super_dict[super_ix][class_ix].append(image_sourcepath + '/' + image_path)

        ### Split the training data into a train/val setup by class.
        if split != "test":
            train_image_dict, val_image_dict = {}, {}
            train_count, val_count = 0, 0
            for super_ix in super_dict.keys():
                class_ixs = sorted(list(super_dict[super_ix].keys()))
                train_val_split = int(len(super_dict[super_ix]) * 0.5)
                train_image_dict[super_ix] = {}
                for _, class_ix in enumerate(class_ixs[:train_val_split]):
                    train_image_dict[super_ix][train_count] = super_dict[super_ix][class_ix]
                    train_count += 1
                val_image_dict[super_ix] = {}
                for _, class_ix in enumerate(class_ixs[train_val_split:]):
                    val_image_dict[super_ix][val_count] = super_dict[super_ix][class_ix]
                    val_count += 1

        ### choose the correct split
        if split == "train":
            train_image_dict_temp = {}
            super_train_image_dict = {}
            train_conversion = {}
            i = 0
            for super_ix, super_set in train_image_dict.items():
                super_ix -= 1
                counter = 0
                super_train_image_dict[super_ix] = []
                for class_ix, class_set in super_set.items():
                    super_train_image_dict[super_ix].extend(class_set)
                    train_image_dict_temp[class_ix] = class_set
                    if class_ix not in train_conversion:
                        train_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                        super_conversion[class_ix] = class_set[0].split('/')[-2]
                    counter += 1
                    i += 1
            train_image_dict = train_image_dict_temp
            self.samples = train_image_dict
        elif split == "validation":
            val_image_dict_temp = {}
            super_val_image_dict = {}
            val_conversion = {}
            i = 0
            for super_ix, super_set in val_image_dict.items():
                super_ix -= 1
                counter = 0
                super_val_image_dict[super_ix] = []
                for class_ix, class_set in super_set.items():
                    super_val_image_dict[super_ix].extend(class_set)
                    val_image_dict_temp[class_ix] = class_set
                    if class_ix not in val_conversion:
                        val_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                        super_conversion[class_ix] = class_set[0].split('/')[-2]
                    counter += 1
                    i += 1
            val_image_dict = val_image_dict_temp
            self.samples = val_image_dict
        elif split == "test" or split == "trainontest" or split == "testontest":
            test_image_dict = {}
            super_test_conversion = {}
            test_conversion = {}
            for class_ix, img_path in zip(test_files['class_id'], test_files['path']):
                class_ix = class_ix - 1
                if not class_ix in test_image_dict.keys():
                    test_image_dict[class_ix] = []
                test_image_dict[class_ix].append(image_sourcepath + '/' + img_path)
                test_conversion[class_ix] = img_path.split('/')[-1].split('_')[0]
                super_test_conversion[class_ix] = img_path.split('/')[-2]
            self.samples = test_image_dict
        # Now flatten all classes into one dict
        self.samples = [[(x, key) for x in self.samples[key]] for key in self.samples.keys()]
        self.samples = [x for y in self.samples for x in y]
        if split == "trainontest":
            min_class_idx = np.min([i[-1] for i in self.samples]).item()
            self.samples = [(self.samples[i][0], self.samples[i][-1] - min_class_idx) for i in np.arange(0, len(self.samples), 2)]
        elif split == "testontest":
            min_class_idx = np.min([i[-1] for i in self.samples]).item()
            self.samples = [(self.samples[i][0], self.samples[i][-1] - min_class_idx) for i in np.arange(1, len(self.samples), 2)]

        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. '
                f'Supported image extensions are {", ".join(get_img_extensions())}')

    def __getitem__(self, index):
        # Return format is
        # 1) PIL image
        # 2) array where first column is target label and remaining columns are raw label counts
        path = self.samples[index][0]
        label = self.samples[index][-1]
        path = os.path.join(self.root, path)
        return pil_loader(path), np.array(label)

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

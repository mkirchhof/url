from torchvision.datasets.folder import ImageFolder
import os
import torch

HIERARCHICAL_TREE = "imagenet21k_miil_tree.pth"

class ImageNet8k(ImageFolder):
    """`ImageNet-8k Classification Dataset. (only classes that have 500+ images and only wordnet leaf nodes)

        Args:
            root (string): Root directory of the ImageNet-21k Dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            reduce_size (bool, optional): Use only every 10th sample (especially interesting for val split)
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            loader (callable, optional): A function to load an image given its path.

         Attributes:
            classes (list): List of the class name tuples.
            class_to_idx (dict): Dict with items (class_name, class_index).
            wnids (list): List of the WordNet IDs.
            wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
            imgs (list): List of (image path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
        """

    def __init__(self, root: str, split: str = 'train', reduce_size=False,  **kwargs) -> None:
        root = self.root = os.path.expanduser(root)
        if split == "train":
            self.split = "imagenet8k_train"
        elif split == "val" or split == "val_small":
            self.split = "imagenet8k_val"
        else:
            raise NotImplementedError(f"ImageNet8k split '{split}' is unknown.")

        # ImageNet8k is simply an ImageFolder
        super(ImageNet8k, self).__init__(self.split_folder, **kwargs)
        self.root = root

        # Match classes to wordnet ids
        # TODO: Match this to imagenet_12k: Give a 8k_class and a val_files.csv file,
        # then create it from the Imagenet-21k directory directly
        self.tree = torch.load(os.path.join(self.root, HIERARCHICAL_TREE))
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [self.tree["class_description"][wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        # TODO: If we want to implement multi-label, then that information is
        # in self.tree["class_tree_list"]. That gives all allowed superclasses of each sample.
        # E.g., class 9 is also 66, 65, and 144 (and class 66 is also 65 and 144).

        if reduce_size:
            every = 10
            self.samples = [item for i, item in enumerate(self.samples) if i % every == 0]
            self.imgs = [item for i, item in enumerate(self.imgs) if i % every == 0]
            self.targets = [item for i, item in enumerate(self.targets) if i % every == 0]

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

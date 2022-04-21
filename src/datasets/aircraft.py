from pathlib import Path

import torch
from PIL import Image
from typing import Callable, Optional
import pickle
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
import json
import numpy as np
from .utils import get_transforms
import collections
from loguru import logger


class Aircraft(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):
        self.transform = get_transforms(args)
        self.target_transform = target_transform

        splits, all_classes = self.create_splits(root)
        # Get the names of the classes assigned to each split
        # Retrieve mapping from filename to bounding box.
        # Cropping to the bounding boxes is important for two reasons:
        # 1) The dataset documentation mentions that "[the] (main) aircraft in each
        #    image is annotated with a tight bounding box [...]", which suggests
        #    that there may be more than one aircraft in some images. Cropping to
        #    the bounding boxes removes ambiguity as to which airplane the label
        #    refers to.
        # 2) Raw images have a 20-pixel border at the bottom with copyright
        #    information which needs to be removed. Cropping to the bounding boxes
        #    has the side-effect that it removes the border.
        bboxes_path = root / "images_box.txt"
        with open(bboxes_path, "r") as f:
            names_to_bboxes = [line.split("\n")[0].split(" ") for line in f.readlines()]
            names_to_bboxes = dict(
                (name, map(int, (xmin, ymin, xmax, ymax)))
                for name, xmin, ymin, xmax, ymax in names_to_bboxes
            )

        # Retrieve mapping from filename to variant
        variant_trainval_path = root / "images_variant_trainval.txt"
        with open(variant_trainval_path, "r") as f:
            names_to_variants = [
                line.split("\n")[0].split(" ", 1) for line in f.readlines()
            ]

        variant_test_path = root / "images_variant_test.txt"
        with open(variant_test_path, "r") as f:
            names_to_variants += [
                line.split("\n")[0].split(" ", 1) for line in f.readlines()
            ]

        names_to_variants = dict(names_to_variants)

        # Build mapping from variant to filenames. "Variant" refers to the aircraft
        # model variant (e.g., A330-200) and is used as the class name in the
        # dataset. The position of the class name in the concatenated list of
        # training, valation, and test class name constitutes its class ID.
        variants_to_names = collections.defaultdict(list)
        for name, variant in names_to_variants.items():
            variants_to_names[variant].append(name)

        self.images = []
        self.bboxes = []
        self.labels = []
        for class_id, class_name in enumerate(all_classes):
            self.images += [
                root / "images" / "{}.jpg".format(filename)
                for filename in sorted(variants_to_names[class_name])
            ]
            self.labels += [class_id] * len(variants_to_names[class_name])
            self.bboxes += [
                names_to_bboxes[name] for name in sorted(variants_to_names[class_name])
            ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img_path, label = (
            self.images[item],
            self.labels[item],
        )
        img = self.load_image(img_path)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def load_image(self, filename):
        return self.transform(Image.open(filename).convert("RGB"))

    def create_splits(self, root):
        """Create splits for Aircraft and store them in the default path.
        If no split file is provided, and the default location for Aircraft splits
        does not contain a split file, splits are randomly created in this
        function using 70%, 15%, and 15% of the data for training, valation and
        testing, respectively, and then stored in that default location.
        Returns:
          The splits for this dataset, represented as a dictionary mapping each of
          'train', 'val', and 'test' to a list of strings (class names).
        """
        NUM_TRAIN_CLASSES = 70
        NUM_val_CLASSES = 15
        NUM_TEST_CLASSES = 15

        train_inds = np.arange(NUM_TRAIN_CLASSES)
        val_inds = NUM_TRAIN_CLASSES + np.arange(NUM_val_CLASSES)
        test_inds = NUM_TRAIN_CLASSES + NUM_val_CLASSES + np.arange(NUM_TEST_CLASSES)
        # "Variant" refers to the aircraft model variant (e.g., A330-200) and is
        # used as the class name in the dataset.
        variants_path = root / "variants.txt"
        with open(variants_path, "r") as f:
            variants = [line.strip() for line in f.readlines() if line]
        variants = sorted(variants)
        assert (
            len(variants) == NUM_TRAIN_CLASSES + NUM_val_CLASSES + NUM_TEST_CLASSES
        ), (len(variants), NUM_TRAIN_CLASSES + NUM_val_CLASSES + NUM_TEST_CLASSES)

        splits = {
            "train": [variants[i] for i in train_inds],
            "val": [variants[i] for i in val_inds],
            "test": [variants[i] for i in test_inds],
        }
        return splits, variants

from pathlib import Path

from loguru import logger
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


class Fungi(VisionDataset):
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

        with open(root / f"train.json", "r") as f:
            original_train = json.load(f)
        with open(root / f"val.json", "r") as f:
            original_val = json.load(f)

        self.images, self.labels = [], []

        # class_splits = self.create_splits(root)

        # Add all images and records
        for image_record, annot_record in zip(
            original_train["images"] + original_val["images"],
            original_train["annotations"] + original_val["annotations"],
        ):

            assert image_record["id"] == annot_record["image_id"], (
                image_record,
                annot_record,
            )
            self.images.append(root / image_record["file_name"])
            self.labels.append(annot_record["category_id"])

        logger.info(f"Fungi {split} loaded. {len(self.images)} images found.")

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

        NUM_TRAIN_CLASSES = 994
        NUM_VALID_CLASSES = 200
        NUM_TEST_CLASSES = 200

        """
        Largely insipired by https://github.com/google-research/meta-dataset/blob/ca81edbf5093ec5ea1a1f5a4b31ec4078825f44b/meta_dataset/dataset_conversion/dataset_to_records.py#L1609
        Create splits for Fungi and store them in the default path.
        If no split file is provided, and the default location for Fungi Identity
        splits does not contain a split file, splits are randomly created in this
        function using 70%, 15%, and 15% of the data for training, validation and
        testing, respectively, and then stored in that default location.
        Returns:
          The splits for this dataset, represented as a dictionary mapping each of
          'train', 'valid', and 'test' to a list of class names.
        """
        # We ignore the original train and validation splits (the test set cannot be
        # used since it is not labeled).
        with open(root / "train.json", "r") as f:
            original_train = json.load(f)
        with open(root / "val.json", "r") as f:
            original_val = json.load(f)

        # The categories (classes) for train and validation should be the same.
        assert original_train["categories"] == original_val["categories"]
        # Sort by category ID for reproducibility.
        categories = sorted(original_train["categories"], key=lambda x: x["id"])

        # Assert contiguous range [0:category_number]
        assert [category["id"] for category in categories] == list(
            range(len(categories))
        )

        # Some categories share the same name (see
        # https://github.com/visipedia/fgvcx_fungi_comp/issues/1)
        # so we include the category id in the label.
        labels = [category["id"] for category in categories]

        train_inds = np.arange(NUM_TRAIN_CLASSES)
        valid_inds = NUM_TRAIN_CLASSES + np.arange(NUM_VALID_CLASSES)
        test_inds = NUM_TRAIN_CLASSES + NUM_VALID_CLASSES + np.arange(NUM_TEST_CLASSES)

        splits = {
            "train": [labels[i] for i in train_inds],
            "val": [labels[i] for i in valid_inds],
            "test": [labels[i] for i in test_inds],
        }

        return splits

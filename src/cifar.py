import json
import os
from pathlib import Path

from PIL import Image
import pickle
from typing import Any

import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class FewShotCIFAR100(CIFAR100):
    def __init__(
        self,
        root: Path,
        specs_file: Path,
        image_size: int = 32,
        training: bool = False,
        download: bool = False,
    ):

        transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            )
        )

        super(CIFAR10, self).__init__(
            str(root),
            transform=transform,
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        downloaded_list = self.train_list + self.test_list

        self._load_meta()

        with open(specs_file, "r") as file:
            self.specs = json.load(file)

        self.class_to_idx = {
            class_name: self.class_to_idx[class_name]
            for class_name in self.specs["class_names"]
        }
        self.sorted_class_ids = sorted(self.class_to_idx.values())

        self.images: Any = []
        self.labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                items_to_keep = [
                    item
                    for item in range(len(entry["data"]))
                    if entry["fine_labels"][item] in self.class_to_idx.values()
                ]
                self.images.append([entry["data"][item] for item in items_to_keep])
                self.labels.extend(
                    [entry["fine_labels"][item] for item in items_to_keep]
                )

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.transform(Image.fromarray(self.images[item]))

        label = self.sorted_class_ids.index(self.labels[item])

        return img, label

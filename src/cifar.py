from functools import partial
import json
import os
from PIL import Image
import pickle
from typing import Any, Callable, Optional

import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


class FewShotCIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        specs_file: str,
        image_size: int = 32,
        training: bool = False,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):

        transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                    transforms.ToTensor(),
                ]
            )
        )

        super(CIFAR10, self).__init__(
            root, transform=transform, target_transform=target_transform
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
        self.id_to_class = {v: k for k, v in self.class_to_idx.items()}

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
        label = self.labels[item]

        return img, label

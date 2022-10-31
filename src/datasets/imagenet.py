from pathlib import Path
from loguru import logger
import torch
from PIL import Image
from typing import Callable, Optional

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import os.path as osp
from .utils import get_transforms


class ImageNet(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):
        super().__init__(str(root), target_transform=target_transform)
        self.transform = get_transforms(args)
        self.target_transform = target_transform
        self.images = []
        self.labels = []
        if split == "train":
            for class_id, class_name in enumerate((root / split).iterdir()):
                for _, _, files in os.walk(root / split / class_name):
                    for file in files:
                        self.labels.append(class_id)
                        self.images.append(root / split / class_name / file)
        else:
            self.images = list((root / split).iterdir())
            self.labels = [0] * len(self.images)

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

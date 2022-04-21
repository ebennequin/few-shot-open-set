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
    """
    Placeholder for compatitiblity
    """

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
        self.images = []
        self.labels = []
        for class_id, class_name in enumerate(os.listdir(root / "val")):
            for _, _, files in os.walk(root / "val" / class_name):
                for file in files:
                    self.labels.append(class_id)
                    self.images.append(root / "val" / class_name / file)

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

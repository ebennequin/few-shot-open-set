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
from .utils import get_normalize


class Aircraft(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):
        NORMALIZE = get_normalize(args)
        self.target_transform = target_transform
        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(args.image_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZE

                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize(int(args.image_size*256/224)),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    NORMALIZE
                ]
            )
        )

        with open(root / 'families.txt', 'r') as f:
            families = f.readlines()
        families = list(map(lambda x: x.strip(), families))
        families.sort()

        with open(root / 'images_family_trainval.txt', 'r') as f:
            image_labels = f.readlines()
        image_labels = list(map(lambda x: x.strip().split(' ', 1), image_labels))
        image_list = list(map(lambda x: x[0], image_labels))
        image_path = list(map(lambda x: Path(root) / 'images' / f"{x}.jpg", image_list))
        labels = list(map(lambda x: families.index(x[1]), image_labels))

        self.images = image_path
        self.labels = labels

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

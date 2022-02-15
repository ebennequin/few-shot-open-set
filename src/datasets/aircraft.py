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


class Aircraft(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):
        self.target_transform = target_transform
        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(args.image_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))

                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize(int(args.image_size*256/224)),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
                ]
            )
        )

        with open(root, 'families.txt', 'r') as f:
            families = f.readlines()
        families = list(map(families, lambda x: x.strip()))
        families.sort()

        with open(root / 'images_family_trainval.txt', 'r') as f:
            image_labels = f.readlines()
        image_labels = list(map(image_labels, lambda x: x.strip()))
        image_list = list(map(image_labels, lambda x: x[0]))
        image_path = list(map(image_list, lambda x: Path(root) / 'images' / f"{x}.jpg"))
        labels = list(map(image_labels, lambda x: families.index(x[1])))

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

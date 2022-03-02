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

class CUB(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        training: bool = False,
    ):
        self.transform = get_transforms(args)

        with open(root / 'images.txt', 'r') as f:
            image_list = f.readlines()

        image_index = []
        image_path = []
        for data in image_list:
            index, path = data.split(' ')
            image_index.append(int(index))
            image_path.append(root / 'images' / path[:-1])

        self.images = image_path

        train_flag = np.loadtxt(root / 'train_test_split.txt', delimiter=' ', dtype=np.int32)
        train_flag = train_flag[:, 1]
        labels = np.loadtxt(root / 'image_class_labels.txt', delimiter=' ', dtype=np.int32)
        labels = labels[:, 1]

        # use first 100 classes
        targets = np.where(labels < 101)[0]
        self.labels = labels
        self.indices = targets
        self.label = list(self.labels[self.indices] - 1)
        self.num_classes = self.num_class = 100

        train_flag = np.loadtxt(root / 'train_test_split.txt', delimiter=' ', dtype=np.int32)
        train_flag = train_flag[:, 1]
        labels = np.loadtxt(root / 'image_class_labels.txt', delimiter=' ', dtype=np.int32)
        labels = labels[:, 1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        img_path, label = (
            self.images[index],
            self.labels[index],
        )
        img = self.load_image(img_path)

        if self.get_transforms is not None:
            label = self.get_transforms(label)

        return img, label

    def load_image(self, filename):
        return self.transform(Image.open(filename).convert("RGB"))

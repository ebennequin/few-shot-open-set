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


class CUB(VisionDataset):
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

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def load_image(self, filename):
        return self.transform(Image.open(filename).convert("RGB"))

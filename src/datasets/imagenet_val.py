from pathlib import Path

import torch
import xmltodict
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


class ImageNetVal(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        target_transform: Optional[Callable] = None,
    ):
        self.transform = get_transforms(args)
        self.target_transform = target_transform
        self.images = sorted((root / "Data/CLS-LOC/val").glob("*"))
        self.annotations = sorted((root / "Annotations/CLS-LOC/val").glob("*"))
        self.class_names = []
        for annotation in self.annotations:
            with open(annotation, "r") as f:
                current_annotation = xmltodict.parse(f.read())["annotation"]["object"]
                if type(current_annotation) == list:
                    class_name = current_annotation[0]["name"]
                else:
                    class_name = current_annotation["name"]
            self.class_names.append(class_name)
        self.class_to_idx = {
            class_name: i for i, class_name in enumerate(set(self.class_names))
        }
        self.labels = [self.class_to_idx[class_name] for class_name in self.class_names]

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

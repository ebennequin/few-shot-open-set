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

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# class TieredImageNet(VisionDataset):
#     def __init__(
#         self,
#         args,
#         root: Path,
#         specs_file: Path,
#         target_transform: Optional[Callable] = None,
#         training: bool = False,
#     ):

#         transform = transforms.Compose([
#                                        transforms.ToTensor(),
#                                        NORMALIZE,
#                                        ]
#                                        )

#         super(TieredImageNet, self).__init__(
#             str(root), transform=transform, target_transform=target_transform
#         )

#         # Get images and labels
#         split = specs_file.stem
#         self.labels = self.load_data_from_pkl(root / f'{split}_labels.pkl')['labels']
#         self.images = np.load(root / f'{split}_images.npz')['images']
#         self.class_list = set(self.labels)
#         self.id_to_class = dict(enumerate(self.class_list))
#         self.class_to_id = {v: k for k, v in self.id_to_class.items()}

#     def __len__(self):
#         return len(self.images)

#     def load_data_from_pkl(self, file):
#         try:
#             with open(file, 'rb') as fo:
#                 data = pickle.load(fo)
#             return data
#         except:
#             with open(file, 'rb') as f:
#                 u = pickle._Unpickler(f)
#                 u.encoding = 'latin1'
#                 data = u.load()
#             return data

#     def __getitem__(self, item):

#         img, label = (
#             self.transform(self.images[item]),
#             self.labels[item],
#         )

#         if self.target_transform is not None:
#             label = self.target_transform(label)

#         return img, label


class TieredImageNet(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):

        transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(args.image_size),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZE,

                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize(int(args.image_size*256/224)),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            )
        )

        images_path = root / 'images'
        super(TieredImageNet, self).__init__(
            str(images_path), transform=transform, target_transform=target_transform
        )

        # Get images and labels
        with open(root / 'specs' / f'{split}.json', "r") as file:
            split_dict = json.load(file)

        self.class_list = split_dict['class_names']
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        images = []
        self.labels = []
        for class_ in tqdm(self.class_list):
            path = images_path / class_
            all_images = path.glob("*.JPEG")
            for img_path in all_images:
                images.append(img_path)
                self.labels.append(self.class_to_id[class_])
        self.images = images

    def __len__(self):
        return len(self.images)

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

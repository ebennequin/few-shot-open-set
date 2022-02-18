from pathlib import Path

import torch
from PIL import Image
from typing import Callable, Optional

import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm

import numpy as np
import os.path as osp
from .utils import get_normalize

# class MiniImageNet(VisionDataset):
#     """ Usage:
#     """
#     def __init__(self,
#                  args,
#                  root: Path,
#                  split: str,
#                  target_transform: Optional[Callable] = None,
#                  training: bool = False,
#                  ):

#         self.IMAGE_PATH = osp.join(root, 'images')
#         self.SPLIT_PATH = osp.join(root, 'splits')

#         csv_path = osp.join(self.SPLIT_PATH, split + '.csv')

#         self.data, self.labels = self.parse_csv(csv_path, split)
#         self.num_class = len(set(self.labels))

#         image_size = 84
#         if training:
#             transforms_list = [
#                   transforms.RandomResizedCrop(image_size),
#                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#                   transforms.RandomHorizontalFlip(),
#                   transforms.ToTensor(),
#                 ]
#         else:
#             transforms_list = [
#                   transforms.Resize(int(256 / 224 * image_size)),
#                   transforms.CenterCrop(image_size),
#                   transforms.ToTensor(),
#                 ]

#         if args.backbone == 'resnet12':
#             self.transform = transforms.Compose(
#                 transforms_list + [
#                     transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
#                                          np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
#                 ])
#         elif args.backbone == 'wrn2810' or args.backbone == 'resnet18':
#             self.transform = transforms.Compose(
#                  transforms_list + [
#                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225]
#                                          )
#                                     ])

#     def parse_csv(self, csv_path, split):
#         lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

#         data = []
#         label = []
#         lb = -1

#         self.wnids = []

#         for l in tqdm(lines, ncols=64):
#             name, wnid = l.split(',')
#             path = osp.join(self.IMAGE_PATH, name)
#             if wnid not in self.wnids:
#                 self.wnids.append(wnid)
#                 lb += 1
#             data.append( path )
#             label.append(lb)

#         return data, label

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, i):
#         data, label = self.data[i], self.labels[i]
#         image = self.transform(Image.open(data).convert('RGB'))
#         return image, label

class MiniImageNet(VisionDataset):
    def __init__(
        self,
        args,
        root: Path,
        split: str,
        target_transform: Optional[Callable] = None,
        training: bool = False,
    ):
        NORMALIZE = get_normalize(args)
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
        image_path = root / 'images'
        super(MiniImageNet, self).__init__(
            str(image_path), transform=transform, target_transform=target_transform
        )

        # Get images and labels
        data_df = pd.read_csv(root / 'specs' / f"{split}_images.csv").assign(
            image_paths=lambda df: df.apply(
                lambda row: image_path / row["class_name"] / row["image_name"], axis=1
            )
        )

        self.images = [
                image_path for image_path in tqdm(data_df.image_paths)
            ]

        self.class_list = data_df.class_name.unique()
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.labels = list(data_df.class_name.map(self.class_to_id))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        img, label = (
            self.load_image(self.images[item]),
            self.labels[item],
        )

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def load_image(self, filename):
        return self.transform(Image.open(filename).convert("RGB"))

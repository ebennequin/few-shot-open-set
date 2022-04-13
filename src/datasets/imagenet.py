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

        self.labels = np.arange(1000)

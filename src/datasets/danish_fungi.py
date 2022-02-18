import pandas as pd
import json
from pathlib import Path
from typing import List, Union, Tuple

from PIL import Image
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from easyfsl.data_tools import EasySet

NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class DanishFungi(EasySet):
    def __init__(
        self,
        root: Union[Path, str] = Path("data/fungi") / "images",
        specs_file: Union[Path, str] = Path("data/fungi/specs") / "DF20_metadata.csv",
        image_size=84,
        training=False,
    ):
        """
        Args:
            root: directory where all the images are
            specs_file: path to the CSV file
            image_size: images returned by the dataset will be square images of the given size
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip.
        """
        self.root = Path(root)
        self.data = self.load_specs(specs_file)

        self.class_names = list(self.data.scientific_name.unique())
        self.labels = list(self.data.label)

        self.transform = self.compose_transforms(image_size, training)

    @staticmethod
    def load_specs(specs_file: Path) -> DataFrame:
        """
        Load specs from a CSV file.
        Args:
            specs_file: path to the CSV file

        Returns:
            curated data contained in the CSV file
        """
        data = pd.read_csv(specs_file)
        class_names = list(data.scientific_name.unique())
        label_mapping = {name: class_names.index(name) for name in class_names}
        return data.assign(label=lambda df: df.scientific_name.map(label_mapping))

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
            The type of the image object depends of the output type of self.transform. By default
            it's a torch.Tensor, however you are free to define any function as self.transform, and
            therefore any type for the output image. For instance, if self.transform = lambda x: x,
            then the output image will be of type PIL.Image.Image.
        """
        img = self.transform(
            Image.open(self.root / self.data.image_path[item]).convert("RGB")
        )
        label = self.data.label[item]

        return img, label

    def __len__(self) -> int:
        return len(self.data)

from pathlib import Path
from typing import Union

from easyfsl.data_tools import EasySet
from torchvision import transforms

from src.constants import NORMALIZE


class TieredImageNet(EasySet):
    def __init__(self, specs_file: Union[Path, str], image_size=84, training=False):
        super().__init__(
            specs_file=specs_file, image_size=image_size, training=training
        )

        self.transform = (
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size),
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            )
            if training
            else transforms.Compose(
                [
                    transforms.Resize([int(image_size * 1.15), int(image_size * 1.15)]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            )
        )

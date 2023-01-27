import numpy as np
from torchvision import transforms
from src.models import BACKBONE_CONFIGS
from loguru import logger
from torchvision.transforms import InterpolationMode

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def get_transforms(args):
    mean = BACKBONE_CONFIGS[args.backbone]["mean"]
    std = BACKBONE_CONFIGS[args.backbone]["std"]
    interp = INTERPOLATIONS[BACKBONE_CONFIGS[args.backbone]["interpolation"]]
    crop_pct = BACKBONE_CONFIGS[args.backbone]["crop_pct"]
    NORMALIZE = transforms.Normalize(mean, std)
    image_size = BACKBONE_CONFIGS[args.backbone]["input_size"][-1]

    if (
        args.tgt_dataset == "tiered_imagenet"
        or args.tgt_dataset == "tiered_imagenet_bis"
    ):
        res = transforms.Compose(
            [
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )
    else:
        res = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation=interp),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )
    return res

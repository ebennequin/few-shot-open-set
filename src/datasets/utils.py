import numpy as np
from torchvision import transforms
from src.models import BACKBONE_CONFIGS


def get_transforms(args):

    mean = BACKBONE_CONFIGS[args.backbone]['mean']
    std = BACKBONE_CONFIGS[args.backbone]['std']
    NORMALIZE = transforms.Normalize(mean, std)
    image_size = BACKBONE_CONFIGS[args.backbone]['input_size'][-1]

    if args.tgt_dataset == 'tiered_imagenet':
        return transforms.Compose(
            [
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )
    else:
        transforms.Compose(
                [
                    transforms.Resize(int(image_size*256/224)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    NORMALIZE,
                ]
            )
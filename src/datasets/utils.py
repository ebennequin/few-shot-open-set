import numpy as np
from torchvision import transforms


def get_normalize(args):
    if args.backbone == 'resnet12' and args.model_source == 'feat':
        NORMALIZE = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
    elif args.backbone == 'vitb16':
        NORMALIZE = transforms.Normalize(0.5, 0.5)
    else:
        NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return NORMALIZE
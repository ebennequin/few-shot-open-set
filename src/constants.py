from pathlib import Path

from easyfsl.methods import PrototypicalNetworks
from torchvision import transforms

from src.resnet import resnet12, resnet18, resnet34, resnet12imagenet, resnet18imagenet

# Data
CIFAR_SPECS_DIR = Path("data") / "cifar" / "specs"
MINI_IMAGENET_SPECS_DIR = Path("data") / "mini_imagenet" / "specs"
TIERED_IMAGENET_SPECS_DIR = Path("data") / "tiered_imagenet" / "specs"

CIFAR_ROOT_DIR = Path("data") / "cifar" / "data"
MINI_IMAGENET_ROOT_DIR = Path("data") / "mini_imagenet" / "images"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Stage outputs
TRAINED_MODELS_DIR = Path("data") / "models"
FEATURES_DIR = Path("data") / "features"
TB_LOGS_DIR = Path("data") / "tb_logs"


# Models
BACKBONES = {
    "resnet12": resnet12,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet12i": resnet12imagenet,
    "resnet18i": resnet18imagenet,
}
FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

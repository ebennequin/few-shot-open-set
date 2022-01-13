from pathlib import Path

from easyfsl.methods import PrototypicalNetworks
from torchvision import transforms

from src.tadam_resnet import tadam_res12
from src.resnet import resnet12, resnet18, resnet34, resnet12imagenet, resnet18imagenet

# Data
DATA_ROOT_DIR = Path("data")

CIFAR_SPECS_DIR = DATA_ROOT_DIR / "cifar" / "specs"
MINI_IMAGENET_SPECS_DIR = DATA_ROOT_DIR / "mini_imagenet" / "specs"
TIERED_IMAGENET_SPECS_DIR = DATA_ROOT_DIR / "tiered_imagenet" / "specs"

CIFAR_ROOT_DIR = DATA_ROOT_DIR / "cifar" / "data"
MINI_IMAGENET_ROOT_DIR = DATA_ROOT_DIR / "mini_imagenet" / "images"

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Stage outputs
TRAINED_MODELS_DIR = DATA_ROOT_DIR / "models"
FEATURES_DIR = DATA_ROOT_DIR / "features"
TB_LOGS_DIR = DATA_ROOT_DIR / "tb_logs"


# Models
BACKBONES = {
    "resnet12": resnet12,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet12i": resnet12imagenet,
    "resnet18i": resnet18imagenet,
    "tadam_res12": tadam_res12,
}
FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

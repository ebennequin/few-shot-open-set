from pathlib import Path

from easyfsl.methods import PrototypicalNetworks

from src.models.resnet import resnet18, resnet34
from src.models.wide_resnet import wrn2810
from src.models.custom_resnet import resnet12

# Data
CIFAR_SPECS_DIR = Path("data") / "cifar" / "specs"
MINI_IMAGENET_SPECS_DIR = Path('/ssd/dataset/natural/original/mini_imagenet/splits')
TIERED_IMAGENET_SPECS_DIR = Path("data") / "tiered_imagenet" / "specs"

CIFAR_ROOT_DIR = Path("data") / "cifar" / "data"
MINI_IMAGENET_ROOT_DIR = Path('/ssd/dataset/natural/original/mini_imagenet')
TIERED_IMAGENET_ROOT_DIR = Path("data") / "tiered_imagenet"


# Stage outputs
TRAINED_MODELS_DIR = Path("data") / "models"
FEATURES_DIR = Path("data") / "features"
TB_LOGS_DIR = Path("data") / "tb_logs"


# Models
BACKBONES = {
    "resnet12": resnet12,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "wrn2810": wrn2810,
}
FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

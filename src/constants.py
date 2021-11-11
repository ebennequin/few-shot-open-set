from pathlib import Path

# Directories
from easyfsl.methods import PrototypicalNetworks

# Data
from src.custom_models import resnet12, resnet18, resnet34

CIFAR_SPECS_DIR = Path("data") / "cifar100" / "specs"
MINI_IMAGENET_SPECS_DIR = Path("data") / "mini_imagenet" / "specs"

CIFAR_ROOT_DIR = Path("data") / "cifar100" / "data"
MINI_IMAGENET_ROOT_DIR = Path("data") / "mini_imagenet" / "images"


# Stage outputs
TRAINED_MODELS_DIR = Path("data") / "models"
TB_LOGS_DIR = Path("data") / "tb_logs"


# Models
BACKBONES = {
    "resnet12": resnet12,
    "resnet18": resnet18,
    "resnet34": resnet34,
}
FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

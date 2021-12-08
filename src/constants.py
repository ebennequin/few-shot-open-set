from pathlib import Path

from easyfsl.methods import PrototypicalNetworks

from src.resnet import resnet12, resnet18, resnet34

# Data
CIFAR_SPECS_DIR = Path("data") / "cifar" / "specs"
MINI_IMAGENET_SPECS_DIR = Path("data") / "mini_imagenet" / "specs"

CIFAR_ROOT_DIR = Path("data") / "cifar" / "data"
MINI_IMAGENET_ROOT_DIR = Path("data") / "mini_imagenet" / "images"


# Stage outputs
TRAINED_MODELS_DIR = Path("data") / "models"
FEATURES_DIR = Path("data") / "features"
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

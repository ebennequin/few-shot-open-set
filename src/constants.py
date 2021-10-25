from pathlib import Path

# Directories
from easyfsl.methods import PrototypicalNetworks
from torchvision.models import resnet18, mobilenet_v3_small, squeezenet1_1, resnet34

CIFAR_SPECS_DIR = Path("data") / "cifar100" / "specs"
TRAINED_MODELS_DIR = Path("data") / "models"
TB_LOGS_DIR = Path("data") / "tb_logs"


# Models
BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "mobilenet": mobilenet_v3_small,
    "squeezenet": squeezenet1_1,
}
FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

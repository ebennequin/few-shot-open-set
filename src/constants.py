from pathlib import Path

from easyfsl.methods import PrototypicalNetworks

from src.models.resnet import resnet18, resnet34
from src.models.wide_resnet import wrn2810
from src.models.custom_resnet import resnet12
from src.models.snatcher_f import SnaTCHerF
from src.models.vit import vit_b16


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
    "vitb16": vit_b16,
}

MISC_MODULES = {
    "snatcher_f": SnaTCHerF
}

FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

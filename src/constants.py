from pathlib import Path

from easyfsl.methods import PrototypicalNetworks

from src.models.snatcher_f import SnaTCHerF


# Stage outputs
TRAINED_MODELS_DIR = Path("data") / "models"
FEATURES_DIR = Path("data") / "features"
TB_LOGS_DIR = Path("data") / "tb_logs"


MISC_MODULES = {
    "snatcher_f": SnaTCHerF
}

FEW_SHOT_METHODS = {
    "protonet": PrototypicalNetworks,
}

from pathlib import Path
from pydoc import locate

import yaml
from loguru import logger

PARAMETERS_FILE = Path("pipelines") / "compute_features" / "params.yaml"


with open(PARAMETERS_FILE) as file:
    logger.info(f'Parsing parameters from "{PARAMETERS_FILE}"')
    params = yaml.safe_load(file)

BACKBONE = locate(params["backbone"])
BATCH_SIZE = params["batch_size"]
N_WORKERS = params["n_workers"]
DEVICE = params["device"]

from pathlib import Path
from pydoc import locate

import yaml
from loguru import logger

PARAMETERS_FILE = Path("pipelines") / "inference" / "params.yaml"
DETECTORS_ROOT = "src.outlier_detection_methods."
CLASSIFIERS_ROOT = "src.few_shot_methods."
TRANSFORMERS_ROOT = "src.feature_transforms."

with open(PARAMETERS_FILE) as file:
    logger.info(f'Parsing parameters from "{PARAMETERS_FILE}"')
    params = yaml.safe_load(file)

RANDOM_SEED = params["random_seed"]
N_WAY = params["n_way"]
N_SHOT = params["n_shot"]
N_QUERY = params["n_query"]
N_TASKS = params["n_tasks"]
N_WORKERS = params["n_workers"]

DETECTOR = locate(DETECTORS_ROOT + params["detector"])
DETECTOR_ARGS = params["detector_args"]
CLASSIFIER = locate(CLASSIFIERS_ROOT + params["classifier"])
CLASSIFIER_ARGS = params["classifier_args"]
PREPOOL_TRANSFORMERS = [
    locate(TRANSFORMERS_ROOT + transformer)
    for transformer in params["prepool_transformers"]
]
POSTPOOL_TRANSFORMERS = [
    locate(TRANSFORMERS_ROOT + transformer)
    for transformer in params["postpool_transformers"]
]
TRANSFORMERS_ARGS = params["transformers_args"]

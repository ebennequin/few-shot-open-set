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

# Problem settings

RANDOM_SEED = params["setting"]["random_seed"]
N_WAY = params["setting"]["n_way"]
N_SHOT = params["setting"]["n_shot"]
N_QUERY = params["setting"]["n_query"]
N_TASKS = params["setting"]["n_tasks"]
N_WORKERS = params["setting"]["n_workers"]

# Problem solvers

DETECTOR = locate(DETECTORS_ROOT + params["method"]["detector"])
DETECTOR_ARGS = params["method"]["detector_args"]
CLASSIFIER = locate(CLASSIFIERS_ROOT + params["method"]["classifier"])
CLASSIFIER_ARGS = params["method"]["classifier_args"]
PREPOOL_TRANSFORMERS = [
    locate(TRANSFORMERS_ROOT + transformer)
    for transformer in params["method"]["prepool_transformers"]
]
POSTPOOL_TRANSFORMERS = [
    locate(TRANSFORMERS_ROOT + transformer)
    for transformer in params["method"]["postpool_transformers"]
]
TRANSFORMERS_ARGS = params["method"]["transformers_args"]

# Metrics
OBJECTIVE = params["objective"]

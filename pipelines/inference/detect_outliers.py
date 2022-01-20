import pickle
import typer
from loguru import logger

from pipelines.inference.params import (
    RANDOM_SEED,
    N_WAY,
    N_SHOT,
    N_QUERY,
    N_TASKS,
    N_WORKERS,
    CLASSIFIER,
    CLASSIFIER_ARGS,
    TRANSFORMERS_ARGS,
    PREPOOL_TRANSFORMERS,
    POSTPOOL_TRANSFORMERS,
    DETECTOR,
    DETECTOR_ARGS,
)
from src.constants import FEATURES_DIR, OUTLIER_PREDICTIONS_CSV
from src.feature_transforms import SequentialFeatureTransformer
from src.utils.data_fetchers import get_test_features, get_features_data_loader
from src.utils.outlier_detectors import detect_outliers
from src.utils.utils import set_random_seed


def main(dataset: str):
    set_random_seed(RANDOM_SEED)

    logger.info(f"Loading features for {dataset}...")
    with open(FEATURES_DIR / f"{dataset}.pickle", "rb") as stream:
        saved_features = pickle.load(stream)

    data_loader = get_features_data_loader(
        features_dict=saved_features["test_set_features"],
        features_to_center_on=saved_features["average_train_set_features"],
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_TASKS,
        n_workers=N_WORKERS,
    )

    logger.info(f"Building model: {DETECTOR.__name__}")
    few_shot_classifier = CLASSIFIER.from_args(CLASSIFIER_ARGS)

    prepool_transformer = SequentialFeatureTransformer(
        [
            transformer.from_args(
                dict(
                    average_train_features=saved_features["average_train_set_features"],
                    **TRANSFORMERS_ARGS,
                )
            )
            for transformer in PREPOOL_TRANSFORMERS
        ]
    )

    postpool_transformer = SequentialFeatureTransformer(
        [
            transformer.from_args(
                dict(
                    average_train_features=saved_features["average_train_set_features"],
                    **TRANSFORMERS_ARGS,
                )
            )
            for transformer in POSTPOOL_TRANSFORMERS
        ]
    )

    outlier_detector = DETECTOR.from_args(
        dict(
            prepool_feature_transformer=prepool_transformer,
            postpool_transformer=postpool_transformer,
            few_shot_classifier=few_shot_classifier,
            **DETECTOR_ARGS,
        )
    )

    logger.info(f"Running inference on {N_TASKS} tasks...")
    outliers_df = detect_outliers(outlier_detector, data_loader, N_WAY, N_QUERY)

    # Saving results
    outliers_df.to_csv(OUTLIER_PREDICTIONS_CSV)
    logger.info(f"Predictions dumped to {OUTLIER_PREDICTIONS_CSV}.")


if __name__ == "__main__":
    typer.run(main)

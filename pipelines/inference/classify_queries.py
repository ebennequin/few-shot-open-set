import pickle

import pandas as pd
import typer
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from src.constants import (
    FEATURES_DIR,
    OUTLIER_PREDICTIONS_CSV,
    CLASSIFICATION_PREDICTIONS_CSV,
    PREDICTIONS_DIR,
)
from src.feature_transforms import SequentialFeatureTransformer
from src.few_shot_methods import AbstractFewShotMethod
from src.utils.data_fetchers import get_test_features, get_features_data_loader
from src.utils.utils import set_random_seed


def main(dataset: str):
    set_random_seed(RANDOM_SEED)

    logger.info(f"Loading features for {dataset}...")
    with open(FEATURES_DIR / f"{dataset}.pickle", "rb") as stream:
        saved_features = pickle.load(stream)

    data_loader = get_features_data_loader(
        features_dict=saved_features["test_set_features"],
        features_to_center_on=None,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=N_TASKS,
        n_workers=N_WORKERS,
    )

    logger.info(f"Building model: {CLASSIFIER.__name__}")

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

    few_shot_classifier = CLASSIFIER.from_args(
        dict(
            prepool_feature_transformer=prepool_transformer,
            postpool_transformer=postpool_transformer,
            **CLASSIFIER_ARGS,
        )
    )

    logger.info(f"Running inference on {N_TASKS} tasks...")
    predictions_df = classify_queries(few_shot_classifier, data_loader, N_WAY, N_QUERY)

    # Saving results
    output_file = PREDICTIONS_DIR / dataset / "classifications.csv"
    output_file.parent.mkdir(exist_ok=True)
    predictions_df.to_csv(output_file)
    logger.info(f"Predictions dumped to {output_file}.")


def classify_queries(
    few_shot_classifier: AbstractFewShotMethod,
    data_loader: DataLoader,
    n_way: int,
    n_query: int,
):
    predictions_df_list = []
    for task_id, (
        support_features,
        support_labels,
        query_features,
        query_labels,
        _,
    ) in tqdm(enumerate(data_loader)):
        _, query_predictions = few_shot_classifier(
            support_features=support_features,
            query_features=query_features,
            support_labels=support_labels,
        )
        predicted_labels = query_predictions.max(dim=1)
        predictions_df_list.append(
            pd.DataFrame(
                {
                    "task": task_id,
                    "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                    "true_label": query_labels,
                    "predicted_label": predicted_labels.indices,
                    "prediction_confidence": predicted_labels.values,
                }
            )
        )

    return pd.concat(predictions_df_list, ignore_index=True)


if __name__ == "__main__":
    typer.run(main)

import json

import pandas as pd
import typer
from loguru import logger
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from pipelines.inference.params import (
    OBJECTIVE,
)
from src.constants import (
    PREDICTIONS_DIR,
)


def main(dataset: str):
    dataset_predictions_dir = PREDICTIONS_DIR / dataset

    # Classification metrics

    classification_predictions_df = pd.read_csv(
        dataset_predictions_dir / "classifications.csv"
    )
    accuracy = (
        (
            classification_predictions_df.true_label
            == classification_predictions_df.predicted_label
        )
        .loc[classification_predictions_df.outlier == False]
        .mean()
    )

    # Outlier detection metrics

    outlier_predictions_df = pd.read_csv(dataset_predictions_dir / "outliers.csv")

    fp_rate, tp_rate, _ = roc_curve(
        outlier_predictions_df.outlier, outlier_predictions_df.outlier_score
    )
    auroc = auc(fp_rate, tp_rate)

    precisions, recalls, _ = precision_recall_curve(
        outlier_predictions_df.outlier, -outlier_predictions_df.outlier_score
    )
    precision_at_recall_objective = precisions[
        next(i for i, value in enumerate(recalls) if value < OBJECTIVE)
    ]
    recall_at_precision_objective = recalls[
        next(i for i, value in enumerate(precisions) if value > OBJECTIVE)
    ]

    metrics = {
        "accuracy": accuracy,
        "auroc": auroc,
        "precision_for_recall": precision_at_recall_objective,
        "recall_for_precision": recall_at_precision_objective,
    }

    metrics_path = dataset_predictions_dir / "metrics.json"
    logger.info(f"Metrics: {json.dumps(metrics, indent=4)}")

    with open(metrics_path, "w") as stream:
        json.dump(metrics, stream, indent=4)
    logger.info(f"Metrics dumped to {metrics_path}.")

    roc_file_path = dataset_predictions_dir / "roc_curve.csv"
    pd.DataFrame(
        {
            "false_positive_rate": fp_rate,
            "true_positive_rate": tp_rate,
        }
    ).to_csv(roc_file_path, index=False)
    logger.info(f"ROC dumped to {roc_file_path}.")


if __name__ == "__main__":
    typer.run(main)

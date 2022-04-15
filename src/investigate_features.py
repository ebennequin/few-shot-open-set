from pathlib import Path

import pickle
import typer
from loguru import logger

from src.utils.plots_and_metrics import clustering_variances_ratio, compute_mean_auroc
from src.utils.utils import normalize


def main(features_path: Path, normalize_features: bool = True):
    logger.info(f"Loading features from {features_path}")
    with open(features_path, "rb") as stream:
        features = pickle.load(stream)
    # WRN returns a weird shape for the features (n_instances, n_channels, 1, 1)
    features = {k: v.reshape(v.shape[0], -1) for k, v in features.items()}
    if normalize_features:
        logger.info("Normalizing features")
        features = normalize(features)

    logger.info(f"Number of classes: {len(features)}")
    logger.info(
        f"Number of images: {sum(len(class_features) for class_features in features.values())}"
    )

    ratio, sigma_within, sigma_between = clustering_variances_ratio(features)
    mean_auroc, average_precision = compute_mean_auroc(features)
    logger.info(f"Metrics:")
    logger.info(f"Intra-class variance: {sigma_within}")
    logger.info(f"Inter-class variance: {sigma_between}")
    logger.info(f"Variance ratio: {ratio}")
    logger.info(f"Mean AUROC: {mean_auroc}")
    logger.info(f"Average Precision: {average_precision}")


if __name__ == "__main__":
    typer.run(main)

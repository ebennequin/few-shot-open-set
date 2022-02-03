import _pickle as cpickle
from collections import OrderedDict

import numpy as np
from pathlib import Path

from loguru import logger
import torch
import typer

from src.constants import (
    FEATURES_DIR,
)
from src.utils.data_fetchers import get_classic_loader
from src.utils.utils import compute_features, load_model


def main(
    backbone: str,
    dataset_name: str,
    weights: Path,
    split: str = "test",
    output_file: Path = None,
    batch_size: int = 256,
    device: str = "cuda",
    layer: str = '4_2',
):
    """
    Compute all features of given dataset images with the given model and dump them in given
    output pickle file.
    Args:
        backbone: what model to train. Must be a key of constants.BACKBONES.
        dataset: what dataset to train the model on.
        weights: path to trained weights
        split: which split of the dataset to infer on
        output_file: where to dump the pickle containing the features
        batch_size: the batch size
        device: what device to train the model on
    """
    logger.info("Fetching data...")
    dataset, data_loader = get_classic_loader(dataset_name, split=split, batch_size=batch_size)

    logger.info("Building model...")
    feature_extractor = load_model(backbone, weights, dataset_name, device)

    logger.info("Computing features...")
    features, labels = compute_features(feature_extractor,
                                        data_loader,
                                        device=device,
                                        split=split,
                                        layer=layer)

    # if output_file is None:
    weights = Path(weights.stem + f'_{layer}').with_suffix(f".pickle").name
    logger.info(weights)
    output_file = FEATURES_DIR / dataset_name / split / weights
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if split == 'test' or split == 'val':
        logger.info("Packing by class...")
        packed_features = {
            class_integer_label: features[labels == class_integer_label]
            for class_integer_label in labels.unique()
        }

        with open(output_file, "wb") as stream:
            cpickle.dump(packed_features, stream, protocol=-1)
    else:
        logger.info("Dumping average feature map...")
        with open(output_file, "wb") as stream:
            cpickle.dump(features, stream, protocol=-1)
    logger.info(f"Dumped features in {output_file}")


if __name__ == "__main__":
    typer.run(main)

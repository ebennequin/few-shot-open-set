import pickle
from collections import OrderedDict

import numpy as np
from pathlib import Path

from loguru import logger
import torch
import typer

from src.constants import (
    BACKBONES,
    FEATURES_DIR,
)
from src.utils import get_classic_loader, compute_features


def main(
    backbone: str,
    dataset: str,
    weights: Path,
    split: str = "test",
    output_file: Path = None,
    batch_size: int = 1024,
    device: str = "cuda",
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
    data_loader = get_classic_loader(dataset, split=split, batch_size=batch_size)

    logger.info("Building model...")
    feature_extractor = BACKBONES[backbone]().to(device)
    feature_extractor.load_state_dict(strip_prefix(torch.load(weights), "backbone."))
    feature_extractor.eval()

    logger.info("Computing features...")
    features, labels = compute_features(feature_extractor, data_loader, device=device)

    logger.info("Packing by class...")
    packed_features = {
        class_integer_label: features[np.where(labels == class_integer_label)]
        for class_integer_label in set(labels)
    }

    if output_file is None:
        output_file = (
            FEATURES_DIR / dataset / split / weights.with_suffix(".pickle").name
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as stream:
        pickle.dump(packed_features, stream, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Dumped features in {output_file}")


def strip_prefix(state_dict: OrderedDict, prefix: str):
    return OrderedDict(
        [
            (k[len(prefix) :] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )


if __name__ == "__main__":
    typer.run(main)

import _pickle as cpickle
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
from src.utils.data_fetchers import get_classic_loader
from src.utils.utils import compute_features


def main(
    backbone: str,
    dataset: str,
    weights: Path,
    split: str = "test",
    output_file: Path = None,
    batch_size: int = 256,
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
    logger.info("Building model...")
    feature_extractor = BACKBONES[backbone]().to(device)
    state_dict = torch.load(weights, map_location=device)
    if "state_dict" in state_dict:
        state_dict = strip_prefix(state_dict["state_dict"], "module.")
    elif "params" in state_dict:
        state_dict = strip_prefix(state_dict["params"], "encoder.")
    else:
        state_dict = strip_prefix(state_dict, "backbone.")

    missing_keys, unexpected = feature_extractor.load_state_dict(state_dict, strict=False)
    print(f"Missing keys {missing_keys}")
    print(f"Unexpected keys {unexpected}")
    feature_extractor.eval()

    logger.info("Fetching data...")
    data_loader = get_classic_loader(dataset, split=split, batch_size=batch_size)

    logger.info("Computing features...")
    features, labels = compute_features(feature_extractor, data_loader, device=device, split=split)

    if output_file is None:
        output_file = FEATURES_DIR / dataset / split / weights.with_suffix(".pickle").name
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


def strip_prefix(state_dict: OrderedDict, prefix: str):
    return OrderedDict(
        [
            (k[len(prefix):] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )


if __name__ == "__main__":
    typer.run(main)
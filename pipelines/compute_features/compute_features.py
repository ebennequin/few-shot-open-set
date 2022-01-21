import pickle
from collections import OrderedDict

import numpy as np
from pathlib import Path

from loguru import logger
import torch
import typer
from torch import nn

from pipelines.compute_features.params import BATCH_SIZE, BACKBONE, DEVICE, N_WORKERS
from src.constants import (
    FEATURES_DIR,
)
from src.utils.data_fetchers import get_classic_loader
from src.utils.utils import compute_features


def main(
    dataset: str,
    weights: Path,
    output_file: Path = None,
):
    """
    Compute all features of given dataset images with the given model and dump them in given
    output pickle file.
    Args:
        dataset: what dataset to train the model on.
        weights: path to trained weights
        output_file: where to dump the pickle containing the features
    """
    logger.info("Building model...")
    feature_extractor = BACKBONE(avg_pool=False).to(DEVICE)
    feature_extractor.load_state_dict(
        strip_prefix(torch.load(weights), "backbone."), strict=False
    )
    feature_extractor.eval()

    logger.info("Starting inference on train set...")
    train_features, _ = infer_on_dataset(
        dataset=dataset, split="train", model=feature_extractor
    )
    average_train_features = train_features.mean(0)

    # On tieredImageNet, training features take ~50Go on RAM, let's make some space
    del train_features

    logger.info("Starting inference on test set...")
    features, labels = infer_on_dataset(
        dataset=dataset, split="test", model=feature_extractor
    )

    logger.info("Packing test set features by class...")
    packed_features = {
        class_integer_label: features[np.where(labels == class_integer_label)]
        for class_integer_label in set(labels)
    }

    if output_file is None:
        output_file = FEATURES_DIR / f"{dataset}.pickle"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as stream:
        pickle.dump(
            {
                "test_set_features": packed_features,
                "average_train_set_features": average_train_features,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    logger.info(f"Dumped features in {output_file}")


def infer_on_dataset(dataset: str, split: str, model: nn.Module):
    logger.info("Fetching data...")
    data_loader = get_classic_loader(
        dataset, split=split, batch_size=BATCH_SIZE, n_workers=N_WORKERS
    )

    logger.info("Computing features...")
    return compute_features(model, data_loader, device=DEVICE)


def strip_prefix(state_dict: OrderedDict, prefix: str):
    return OrderedDict(
        [
            (k[len(prefix) :] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )


if __name__ == "__main__":
    typer.run(main)

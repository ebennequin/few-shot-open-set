import _pickle as cpickle
from collections import OrderedDict

import numpy as np
from pathlib import Path

from loguru import logger
from src.constants import (
    FEATURES_DIR,
    TRAINED_MODELS_DIR
)
from src.utils.data_fetchers import get_classic_loader
from src.utils.utils import compute_features, load_model
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mini_imagenet")
    parser.add_argument("--backbone", type=str, default="resnet12")
    parser.add_argument("--model_source", type=str, default="feat")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--layers", type=str, nargs='+')
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


def main(args):
    weights = TRAINED_MODELS_DIR / args.training / f"{args.backbone}_{args.dataset}_{args.model_source}.pth"
    logger.info("Fetching data...")
    dataset, data_loader = get_classic_loader(args,
                                              dataset_name=args.dataset,
                                              split=args.split,
                                              batch_size=args.batch_size)

    logger.info("Building model...")
    feature_extractor = load_model(args, args.backbone, weights, args.dataset, args.device)

    logger.info("Computing features...")
    features, labels = compute_features(feature_extractor,
                                        data_loader,
                                        device=args.device,
                                        split=args.split,
                                        layers=args.layers)

    # if output_file is None:
    for layer in features:
        pickle_name = Path(weights.stem + f'_{layer}').with_suffix(f".pickle").name
        output_file = FEATURES_DIR / args.dataset / args.split / args.training / pickle_name
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if args.split == 'test' or args.split == 'val':
            logger.info("Packing by class...")
            packed_features = {
                class_integer_label: features[layer][labels == class_integer_label]
                for class_integer_label in labels.unique()
            }
            with open(output_file, "wb") as stream:
                cpickle.dump(packed_features, stream, protocol=-1)
        else:
            logger.info("Dumping average feature map...")
            with open(output_file, "wb") as stream:
                cpickle.dump(features[layer], stream, protocol=-1)
        logger.info(f"Dumped features in {output_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

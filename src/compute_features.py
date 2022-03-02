import _pickle as cpickle

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

    parser.add_argument("--src_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--tgt_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--backbone", type=str, default="resnet12")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_source", type=str, default="feat")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--layers", type=str, nargs='+')
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


def main(args):
    logger.info("Fetching data...")
    dataset, _, data_loader = get_classic_loader(args,
                                                 dataset_name=args.tgt_dataset,
                                                 split=args.split,
                                                 batch_size=args.batch_size)

    logger.info("Building model...")
    if args.model_source == 'url':
        weights = None
        stem = f"{args.backbone}_{args.src_dataset}_{args.model_source}.pth" # used for saving features downstream
    else:
        weights = TRAINED_MODELS_DIR / args.training / f"{args.backbone}_{args.src_dataset}_{args.model_source}.pth"
        stem = weights.stem
    feature_extractor = load_model(args, args.backbone, weights, args.src_dataset, args.device)
    args.layers = feature_extractor.all_layers if args.layers == ['all'] else args.layers
    logger.warning(args.layers)

    logger.info("Computing features...")
    features, labels = compute_features(feature_extractor,
                                        data_loader,
                                        device=args.device,
                                        split=args.split,
                                        layers=args.layers)

    # if output_file is None:
    for layer in features:
        pickle_name = Path(stem + f'_{layer}').with_suffix(f".pickle").name
        output_file = FEATURES_DIR / args.src_dataset / args.tgt_dataset / args.split / args.training / pickle_name
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

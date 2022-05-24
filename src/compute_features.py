import _pickle as cpickle
from pathlib import Path
from src.models import __dict__ as BACKBONES
from loguru import logger
from src.utils.data_fetchers import get_classic_loader
from src.utils.utils import compute_features, load_model
import argparse
from .inference import str2bool

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--tgt_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--backbone", type=str, default="resnet12")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_source", type=str, default="feat")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--override", type=str2bool, default="False")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--keep_all_train_features", type=bool, default=False)
    parser.add_argument("--debug", type=str2bool, default="False")

    args = parser.parse_args()
    return args


def main(args):
    logger.info("Fetching data...")
    dataset, _, data_loader = get_classic_loader(
        args,
        dataset_name=args.tgt_dataset,
        split=args.split,
        batch_size=args.batch_size,
    )

    logger.info("Building model...")
    if args.model_source == "timm":
        weights = None
        stem = f"{args.backbone}_{args.src_dataset}_{args.model_source}"  # used for saving features downstream
    else:
        weights = (
            Path(args.data_dir)
            / "models"
            / args.training
            / f"{args.backbone}_{args.src_dataset}_{args.model_source}.pth"
        )
        stem = weights.stem
    feature_extractor = load_model(
        args, args.backbone, weights, args.src_dataset, args.device
    )

    pickle_name = Path(stem).with_suffix(f".pickle").name
    output_file = (
        Path("data")
        / "features"
        / args.src_dataset
        / args.tgt_dataset
        / args.split
        / args.training
        / pickle_name
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # First checking whether those features already exist
    if output_file.exists():
        logger.info(f"File {output_file} already exists.")
        if args.override:
            logger.warning("Overriding.")
        else:
            logger.warning("Not overriding.")
            return
    else:
        logger.info(f"File {output_file} does not exist. Performing extraction.")
    logger.info("Computing features...")
    features, labels = compute_features(
        feature_extractor,
        data_loader,
        device=args.device,
        split=args.split,
        keep_all_train_features=args.keep_all_train_features,
        debug=args.debug
    )

    # if output_file is None:

    if args.split == "test" or args.split == "val" or args.keep_all_train_features:
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
    args = parse_args()
    main(args)

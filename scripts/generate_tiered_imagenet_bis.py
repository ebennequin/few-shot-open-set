import json
from pathlib import Path

import cv2
import numpy as np
import pickle
import typer
from loguru import logger


from src.utils.imagenet_val_utils import list_imagenet_val

TIERED_IMAGENET_ROOT = Path("data/tiered_imagenet")
TIERED_IMAGENET_SPECS_ROOT = TIERED_IMAGENET_ROOT / "specs"
TIERED_IMAGENET_BIS_ROOT = TIERED_IMAGENET_ROOT / "bis"


def main(ilsvrc_root_dir: Path = Path("/data/etienneb/ILSVRC2015/")):

    logger.info("Reading specs...")
    tiered_imagenet_classes = {}
    for split in ["train", "val", "test"]:
        with open(TIERED_IMAGENET_SPECS_ROOT / f"{split}.json") as f:
            tiered_imagenet_classes[split] = json.load(f)["class_names"]

    image_paths, class_names = list_imagenet_val(ilsvrc_root_dir)

    logger.info("Writing tieredImageNet bis images")
    for split, split_classes in tiered_imagenet_classes.items():
        logger.info(f"Writing {split}")
        images = []
        labels = []
        for integer, class_name in enumerate(split_classes):
            class_images = image_paths[class_names == class_name]
            labels += len(class_images) * [integer]
            for image_path in class_images:
                img = cv2.imread(image_path)
                img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
                images.append(img)

        # Create save_dir
        TIERED_IMAGENET_BIS_ROOT.mkdir(exist_ok=True, parents=True)

        # Save labels
        labels_file = TIERED_IMAGENET_BIS_ROOT / f"{split}_labels.pkl"
        logger.info(f"Saving {len(labels)} labels to {labels_file}...")
        with open(labels_file, "wb") as f:
            pickle.dump({"labels": labels}, f)

        # Save images
        images_file = TIERED_IMAGENET_BIS_ROOT / f"{split}_images.npz"
        logger.info(f"Saving {len(images)} images to {images_file}...")
        np.savez(images_file, images=np.stack(images))


if __name__ == "__main__":
    typer.run(main)

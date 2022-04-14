from pathlib import Path

import typer
from loguru import logger

import shutil
import xmltodict

import pandas as pd

MINI_IMAGENET_SPECS_ROOT = Path("data/mini_imagenet/specs")
MINI_IMAGENET_BIS_ROOT = Path("data/mini_imagenet/bis/")


def main(ilsvrc_root_dir: Path = Path("/data/etienneb/ILSVRC2015/")):

    logger.info("Reading specs...")
    mini_imagenet_classes = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "class_name": pd.read_csv(
                        MINI_IMAGENET_SPECS_ROOT / f"{split}_images.csv"
                    ).class_name.unique(),
                }
            )
            for split in ["train", "val", "test"]
        ]
    )

    image_paths = sorted((ilsvrc_root_dir / "Data/CLS-LOC/val").glob("*"))
    annotations = sorted((ilsvrc_root_dir / "Annotations/CLS-LOC/val").glob("*"))
    class_names = []
    for annotation in annotations:
        with open(annotation, "r") as f:
            current_annotation = xmltodict.parse(f.read())["annotation"]["object"]
            if type(current_annotation) == list:
                class_name = current_annotation[0]["name"]
            else:
                class_name = current_annotation["name"]
        class_names.append(class_name)

    logger.info("Copying miniImageNet bis images")
    image_names_by_class = {}
    for i, image_path in enumerate(image_paths):
        class_name = class_names[i]
        if class_name in set(mini_imagenet_classes.class_name):
            (MINI_IMAGENET_BIS_ROOT / "images" / class_name).mkdir(
                parents=True, exist_ok=True
            )
            shutil.copy(
                image_path,
                (MINI_IMAGENET_BIS_ROOT / "images" / class_name / image_path.name),
            )
            if class_name not in image_names_by_class:
                image_names_by_class[class_name] = []
            image_names_by_class[class_name].append(image_path.name)

    logger.info("Writing CSV specs files")
    images_df = pd.DataFrame(image_names_by_class).melt(
        var_name="class_name", value_name="image_name"
    )

    mini_imagenet_bis_specs = mini_imagenet_classes.merge(images_df, on="class_name")

    (MINI_IMAGENET_BIS_ROOT / "specs").mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        mini_imagenet_bis_specs[mini_imagenet_bis_specs.split == split].drop(
            "split", axis=1
        ).to_csv(MINI_IMAGENET_BIS_ROOT / "specs" / f"{split}_images.csv", index=False)


if __name__ == "__main__":
    typer.run(main)

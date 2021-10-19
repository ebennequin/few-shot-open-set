from pathlib import Path

import click
import pandas as pd
from loguru import logger

from easyfsl.data_tools import EasySemantics, EasySet


@click.option(
    "--split",
    help="What split of the dataset to work on",
    type=click.Choice(["train", "val", "test"]),
    required=True,
)
@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
    type=Path,
    required=True,
)
@click.option(
    "--output-dir",
    help="Where to dump class-distances matrix",
    type=Path,
    required=True,
)
@click.command()
def main(split: str, specs_dir: Path, output_dir: Path):

    logger.info("Creating dataset...")
    train_set = EasySet(specs_file=specs_dir / f"{split}.json", training=False)
    semantic_tools = EasySemantics(train_set, Path(specs_dir / "wordnet.is_a.txt"))

    logger.info("Computing semantic distances...")
    semantic_distances_df = pd.DataFrame(semantic_tools.get_semantic_distance_matrix())

    semantic_distances_df.to_csv(output_dir / f"{split}.csv", index=False, header=False)


if __name__ == "__main__":
    main()

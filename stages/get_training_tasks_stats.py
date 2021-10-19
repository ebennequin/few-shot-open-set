import pickle
from pathlib import Path

import click
import pandas as pd
import seaborn
from loguru import logger
from matplotlib import pyplot as plt

from src.utils import (
    get_intra_class_distances,
    get_sampled_together,
    get_training_confusion,
)


@click.option(
    "--training-tasks-record",
    help="Where to find the record of training tasks",
    type=Path,
    required=True,
)
@click.option(
    "--distances-dir",
    help="Where to find class-distances matrix",
    type=Path,
    required=True,
)
@click.option(
    "--metrics-dir",
    help="Where to find and dump evaluation metrics",
    type=Path,
    required=True,
)
@click.command()
def main(training_tasks_record: Path, distances_dir: Path, metrics_dir: Path):
    logger.info("Loading training records...")
    with open(training_tasks_record, "rb") as file:
        training_tasks_record = pickle.load(file)

    logger.info("Computing metrics...")
    distances = pd.read_csv(distances_dir / "train.csv", header=None).values

    tasks_df = get_intra_class_distances(training_tasks_record, distances)

    n_classes = len(distances)
    global_confusion = get_training_confusion(tasks_df, n_classes)
    biconfusion = (global_confusion + global_confusion.T).triu(diagonal=1)

    sampled_together = get_sampled_together(tasks_df, n_classes)

    tasks_df[["median_distance", "std_distance"]].to_csv(
        metrics_dir / "intra_training_task_distances.csv"
    )

    seaborn.heatmap(sampled_together, linewidth=0)
    plt.savefig(metrics_dir / "training_classes_sampled_together.png")
    plt.clf()
    seaborn.heatmap(biconfusion, linewidth=0)
    plt.savefig(metrics_dir / "training_classes_biconfusion.png")

    logger.info(f"Metrics dumped in {metrics_dir}")


if __name__ == "__main__":
    main()

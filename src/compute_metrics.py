import json
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from easyfsl.utils import get_accuracies


@click.option(
    "--testbed",
    help="Path to the CSV defining the testbed",
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
def main(testbed: Path, metrics_dir: Path):
    results = pd.read_csv(metrics_dir / "raw_results.csv", index_col=0)

    statistics = pd.concat(
        [
            pd.read_csv(testbed, index_col=0).groupby("task").variance.mean(),
            get_accuracies(results),
        ],
        axis=1,
    )

    stats_file = metrics_dir / "task_performances.csv"
    statistics.to_csv(stats_file)
    logger.info(f"Task statistics dumped at {stats_file}")

    metrics_json = metrics_dir / "evaluation_metrics.json"
    with open(metrics_json, "w") as file:
        json.dump(
            {
                "accuracy": statistics.accuracy.mean(),
                "std": statistics.accuracy.std(),
                "first_quartile_acc": statistics.loc[
                    statistics.variance < statistics.variance.quantile(0.25)
                ].accuracy.mean(),
                "second_quartile_acc": statistics.loc[
                    statistics.variance.between(
                        statistics.variance.quantile(0.25),
                        statistics.variance.quantile(0.50),
                    )
                ].accuracy.mean(),
                "third_quartile_acc": statistics.loc[
                    statistics.variance.between(
                        statistics.variance.quantile(0.50),
                        statistics.variance.quantile(0.75),
                    )
                ].accuracy.mean(),
                "fourth_quartile_acc": statistics.loc[
                    statistics.variance.quantile(0.75) <= statistics.variance
                ].accuracy.mean(),
            },
            file,
            indent=4,
        )
    logger.info(f"Metrics dumped to {metrics_json}")

    plot_file = metrics_dir / "accuracy_v_variance.png"
    statistics.plot(x="variance", y="accuracy", kind="scatter")
    plt.savefig(plot_file)
    logger.info(f"Accuracy as a function of task pseudo-variance dumped at {plot_file}")


if __name__ == "__main__":
    main()

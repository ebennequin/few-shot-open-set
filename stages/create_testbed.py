from pathlib import Path
from typing import Dict, List

import click
import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers.utils import sample_label_from_potential
from easyfsl.utils import fill_diagonal, sort_items_per_label
from src.utils import get_pseudo_variance

distances_dir = Path("data/tiered_imagenet/distances")
specs_file = Path("data/tiered_imagenet/specs/test.json")
N_TASKS = 8000
N_WAY = 5
N_SHOT = 5
N_QUERY = 10

ALPHA = 0.3830
BETA_PENALTY = 100.0


@click.option(
    "--n-tasks",
    help="Number of tasks to sample",
    type=int,
    default=5000,
)
@click.option(
    "--n-way",
    help="Number of classes in each task",
    type=int,
    default=5,
)
@click.option(
    "--n-shot",
    help="Number of support images per class",
    type=int,
    default=5,
)
@click.option(
    "--n-query",
    help="Number of query images per class",
    type=int,
    default=10,
)
@click.option(
    "--distances-csv",
    help="Path to the csv containing the distance matrix",
    type=Path,
    default=Path("data/tiered_imagenet/distances/test.csv"),
)
@click.option(
    "--specs-json",
    help="Path to the JSON containing the specs of the test set",
    type=Path,
    default=Path("data/tiered_imagenet/specs/test.json"),
)
@click.option(
    "--alpha",
    help="Weights the importance of the distance in the task sampling."
    "Bigger alpha means more fine-grained tasks.",
    type=float,
    default=0.3830,
)
@click.option(
    "--beta-penalty",
    help="Weights the importance of the frequence-based penalty in the task sampling."
    "Bigger beta means forces the balance between classes in the testbed.",
    type=float,
    default=100.0,
)
@click.option(
    "--seed",
    help="Random seed.",
    type=int,
    default=0,
)
@click.option(
    "--out-file",
    help="Path to the csv where the test bed will be dumped",
    type=Path,
    required=True,
)
@click.command()
def main(
    n_tasks: int,
    n_way: int,
    n_shot: int,
    n_query: int,
    distances_csv: Path,
    specs_json: Path,
    alpha: float,
    beta_penalty: float,
    seed: int,
    out_file: Path,
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    distances = torch.tensor(pd.read_csv(distances_csv, header=None).values)
    logger.info(f"Sampling classes for {n_tasks} tasks...")

    tasks = sample_tasks(n_tasks, n_way, distances, alpha, beta_penalty)

    save_tasks_plots(tasks, out_file)

    logger.info("Sampling images ...")
    tasks_with_samples = sample_items_from_classes(tasks, specs_json, n_shot, n_query)

    tasks_with_samples.to_csv(out_file)
    logger.info(f"Testbed dumped to {out_file}")


def sample_tasks(
    n_tasks: int, n_way: int, distances: torch.Tensor, alpha: float, beta_penalty: float
) -> pd.DataFrame:
    """
    Sample n_tasks tasks, each of which is defined by n_way classes. Sampling is performed according
    to two criterias:
        - we enforce that classes close to each other (according to the specified distances matrix)
        are more likely to be sampled together (the hardness of this criteria is controlled by alpha)
        - we enforce that classes that were already sampled in previous tasks are less likely to be
        sampled again (the hardness of this criteria is controlled by beta_penalty).
    Note that we "oversample": we first generate 2 * n_tasks tasks, then drop duplicate tasks and
    subsample n_tasks tasks from the remaining tasks.
    Args:
        n_tasks: number of output tasks
        n_way: number of classes per task
        distances: matrix of distances between classes. Also defines the total number of classes.
        alpha: hardness of the distance criteria
        beta_penalty: hardness of the penalty for previously sampled classes

    Returns:
        a dataframe with n_tasks * n_way lines, and three columns "task", "labels", and "variance".
        "variance" is defined as the sum of all square distances between pairs of classes in the
        task (and therefore is the same between any two lines that have the same value in the "task"
        column).
    """
    list_of_task_classes = []
    variance = []
    n_appearances = torch.ones((len(distances),))
    potential_matrix = fill_diagonal(torch.exp(-alpha * distances), 0)

    sursampling_factor = 2

    for _ in tqdm(range(sursampling_factor * n_tasks)):
        penalty = torch.exp(-beta_penalty * n_appearances / max(n_appearances))
        task_classes = [sample_label_from_potential(penalty)]
        potential = potential_matrix[task_classes[0]] * penalty

        for _ in range(1, n_way):
            task_classes.append(sample_label_from_potential(potential))
            potential = potential * potential_matrix[task_classes[-1]]

        task_classes = sorted(task_classes)
        for label in task_classes:
            n_appearances[label] += 1.0

        list_of_task_classes.append(task_classes)
        variance.append(get_pseudo_variance(task_classes, distances.numpy()))

    return (
        pd.DataFrame(list_of_task_classes)
        .assign(variance=variance)
        .drop_duplicates(set(range(N_WAY)))
        .sort_values("variance")
        .sample(n_tasks)
        .reset_index(drop=True)
        .melt(id_vars=["variance"], value_name="labels", ignore_index=False)
        .sort_index()
        .drop(columns="variable")
        .reset_index()
        .rename(columns={"index": "task"})
    )


def save_tasks_plots(tasks: pd.DataFrame, out_file: Path):
    """
    Plot two histograms:
        - label occurrences: to evaluate the balance between classes in the testbed.
        - pseudo-variances: to evaluate that the testbed covers quasi-evenly a wide range of
            pseudo-variances
    Args:
        tasks: dataframe of tasks, output of sample_tasks
        out_file: where the testbed will be dumped. We will save the plots next to it.
    """

    tasks.groupby("task").variance.mean().hist().set_title(
        "histogram of pseudo-variances"
    )
    pseudo_variances_file = out_file.parent / f"{out_file.stem}_pv.png"
    plt.savefig(pseudo_variances_file)

    plt.clf()
    tasks.labels.value_counts().hist().set_title("histogram of label occurrences")
    occurrences_file = out_file.parent / f"{out_file.stem}_occ.png"
    plt.savefig(occurrences_file)

    logger.info(f"Histogram of pseudo-variances dumped to {pseudo_variances_file}")
    logger.info(f"Histogram of label occurrences dumped to {occurrences_file}")


class ItemsSampler:
    """
    This is a custom sampler with two necessary properties:
        - sample batchs of elements by label without replacement
        - if there are not enough elements left to constitute a batch, reshuffle labels and restart
    Note: this might be doable with built-in torch samplers (BatchSampler and SubsetRandomSampler).
    """

    def __init__(self, items_per_label: Dict[int, List[int]]):
        self.items_per_label = items_per_label
        self.available_items_per_label = {
            label: items.copy() for label, items in self.items_per_label.items()
        }
        for items in self.available_items_per_label.values():
            np.random.shuffle(items)

    def sample_items(self, n_items: int, label: int) -> List[int]:
        """
        Sample n_times indices from the available indices for the specified labels. If we don't have
        enough indices left, we reshuffle the whole list of indices for this label, and use this as
        the list of available indices.
        Args:
            n_items: number of indices to sample
            label: label for which we sample indices

        Returns:
            n_items integers
        """
        if n_items > len(self.available_items_per_label[label]):
            self.available_items_per_label[label] = self.items_per_label[label].copy()
            np.random.shuffle(self.available_items_per_label[label])

        items = self.available_items_per_label[label][:n_items]
        del self.available_items_per_label[label][:n_items]

        return items


def sample_items_from_classes(
    tasks_df: pd.DataFrame, specs_json: Path, n_shot: int, n_query: int
) -> pd.DataFrame:
    """
    For each label in a task, sample n_shot + n_query images and split them between support and
    query. Images are sampled without replacement to ensure that we use as many images as possible
    for each class.
    Args:
        tasks_df: input dataframe with the columns task, labels and variance
        specs_json: specs file of the test dataset
        n_shot: number of support images per class per task
        n_query: number of query images per class per task

    Returns:
        input dataframe, but with (n_shot + n_query) times more rows, and two more columns:
            image_id (integer) and support (bool).
    """
    test_set = EasySet(specs_json)
    items_per_label = sort_items_per_label(test_set.labels)
    item_sampler = ItemsSampler(items_per_label)

    return (
        tasks_df.assign(
            image_ids=lambda df: [
                item_sampler.sample_items(n_shot + n_query, label)
                for label in df.labels
            ]
        )
        .assign(
            support=lambda df: [items[:n_shot] for items in df.image_ids],
            query=lambda df: [items[n_shot:] for items in df.image_ids],
        )
        .drop(columns="image_ids")
        .melt(
            id_vars=["task", "variance", "labels"],
            value_name="image_id",
            ignore_index=False,
        )
        .explode("image_id")
        .assign(support=lambda df: df.variable == "support")
        .drop(columns="variable")
        .sort_values(["task", "labels", "support"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()

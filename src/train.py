import pickle
from pathlib import Path
from typing import Optional

import click
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.methods import PrototypicalNetworks
from src.utils import build_model, create_dataloader, get_sampler, set_random_seed

SAMPLERS = [
    "uniform",
    "adaptive",
    "semantic",
]


@click.option(
    "--sampler",
    help="How to sample training tasks",
    type=click.Choice(SAMPLERS),
    default="uniform",
)
@click.option(
    "--n-way",
    help="Number of classes per task",
    type=int,
    default=5,
)
@click.option(
    "--n-shot",
    help="Number of support examples per class",
    type=int,
    default=5,
)
@click.option(
    "--n-query",
    help="Number of query samples per class",
    type=int,
    default=10,
)
@click.option(
    "--n-epochs",
    help="Number of training epochs",
    type=int,
    default=100,
)
@click.option(
    "--n-tasks-per-epoch",
    help="Number of episodes per training epoch",
    type=int,
    default=500,
)
@click.option(
    "--semantic-alpha",
    help="Weight of semantic distances for class sampling",
    type=float,
    default=0.5,
)
@click.option(
    "--semantic-strategy",
    help="Curriculum strategy of the semantic sampling",
    type=str,
    default="constant",
)
@click.option(
    "--adaptive-forgetting",
    help="Forgetting hyperparameter for adaptive sampling",
    type=float,
    default=0.5,
)
@click.option(
    "--adaptive-hardness",
    help="Hardness hyperparameter for adaptive sampling",
    type=float,
    default=0.5,
)
@click.option(
    "--pretrained-weights",
    help="It's possible to load pretrained weights. "
    "This ensures that different algorithms have the same starting weights.",
    type=Path,
    required=False,
)
@click.option(
    "--specs-dir",
    help="Where to find the dataset specs files",
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
@click.option(
    "--tb-log-dir",
    help="Where to dump tensorboard event files",
    type=Path,
    required=True,
)
@click.option(
    "--output-model",
    help="Where to dump the archive containing trained model weights",
    type=Path,
    required=True,
)
@click.option(
    "--random-seed",
    help="Defined random seed, for reproducibility",
    type=int,
    default=0,
)
@click.option(
    "--device",
    help="What device to train the model on",
    type=str,
    default="cuda",
)
@click.command()
def main(
    sampler: str,
    n_way: int,
    n_shot: int,
    n_query: int,
    n_epochs: int,
    n_tasks_per_epoch: int,
    semantic_alpha: float,
    semantic_strategy: str,
    adaptive_forgetting: float,
    adaptive_hardness: float,
    pretrained_weights: Optional[Path],
    specs_dir: Path,
    distances_dir: Path,
    metrics_dir: Path,
    tb_log_dir: Path,
    output_model: Path,
    random_seed: int,
    device: str,
):
    metrics_dir.mkdir(parents=True, exist_ok=True)
    n_validation_tasks = 100
    n_workers = 12

    set_random_seed(random_seed)

    logger.info("Fetching training data...")
    train_set = EasySet(specs_file=specs_dir / "train.json", training=True)
    train_sampler = get_sampler(
        sampler=sampler,
        dataset=train_set,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks_per_epoch,
        distances_csv=distances_dir / "train.csv",
        semantic_alpha=semantic_alpha,
        semantic_strategy=semantic_strategy,
        adaptive_forgetting=adaptive_forgetting,
        adaptive_hardness=adaptive_hardness,
    )
    train_loader = create_dataloader(train_set, train_sampler, n_workers)

    logger.info("Fetching validation data...")
    val_set = EasySet(specs_file=specs_dir / "val.json", training=True)
    val_sampler = get_sampler(
        sampler="uniform",
        dataset=val_set,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_validation_tasks,
    )
    val_loader = create_dataloader(val_set, val_sampler, n_workers)

    tb_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building model...")
    model = build_model(
        device=device,
        tb_writer=SummaryWriter(log_dir=tb_log_dir),
        pretrained_weights=pretrained_weights,
    )

    optimizer = Adam(params=model.parameters())

    logger.info("Starting training...")
    training_tasks_record = model.fit_multiple_epochs(
        train_loader,
        optimizer,
        n_epochs=n_epochs,
        val_loader=val_loader,
    )

    record_dump_path = metrics_dir / "training_tasks.pkl"

    with open(record_dump_path, "wb") as file:
        pickle.dump(training_tasks_record, file)

    logger.info(f"Training tasks record dumped at {record_dump_path}")

    torch.save(model.state_dict(), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


if __name__ == "__main__":
    main()

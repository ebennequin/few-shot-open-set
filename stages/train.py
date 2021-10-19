import pickle
from pathlib import Path
from typing import Optional

import click
import torch
from loguru import logger
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from easyfsl.data_tools import EasySet, TaskSampler
from src.utils import build_model, create_dataloader, get_sampler, set_random_seed


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
    n_way: int,
    n_shot: int,
    n_query: int,
    n_epochs: int,
    n_tasks_per_epoch: int,
    pretrained_weights: Optional[Path],
    specs_dir: Path,
    tb_log_dir: Path,
    output_model: Path,
    random_seed: int,
    device: str,
):
    n_validation_tasks = 100
    n_workers = 12

    set_random_seed(random_seed)

    logger.info("Fetching training data...")
    train_set = EasySet(specs_file=specs_dir / "train.json", training=True)
    train_sampler = TaskSampler(
        dataset=train_set,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks_per_epoch,
    )
    train_loader = create_dataloader(train_set, train_sampler, n_workers)

    logger.info("Fetching validation data...")
    val_set = EasySet(specs_file=specs_dir / "val.json", training=True)
    val_sampler = TaskSampler(
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
        pretrained_weights=pretrained_weights,
    )

    optimizer = Adam(params=model.parameters())

    logger.info("Starting training...")
    model.fit_multiple_epochs(
        train_loader,
        optimizer,
        n_epochs=n_epochs,
        val_loader=val_loader,
    )

    torch.save(model.state_dict(), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


if __name__ == "__main__":
    main()

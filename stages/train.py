from pathlib import Path
from typing import Optional

import torch
import typer
from loguru import logger
from torch.optim import Adam

from easyfsl.data_tools import EasySet, TaskSampler

from src.constants import CIFAR_SPECS_DIR, TRAINED_MODELS_DIR
from src.utils import build_model, create_dataloader, set_random_seed


def main(
    backbone: str,
    method: str = "protonet",
    specs_dir: Path = CIFAR_SPECS_DIR,
    output_model: Path = TRAINED_MODELS_DIR / "trained_episodic.tar",
    n_way: int = 5,
    n_shot: int = 5,
    n_query: int = 5,
    n_epochs: int = 100,
    n_tasks_per_epoch: int = 500,
    random_seed: int = 0,
    device: str = "cuda",
    pretrained_weights: Optional[Path] = None,
):
    """
    Train a model in an episodic fashion.
    Args:
        backbone: what model to train. Must be a key of constants.BACKBONES.
        method: what few-shot method to use during episodic training.
            Must be a key of constants.FEW_SHOT_METHODS.
        specs_dir: where to find the dataset specs files
        output_model: where to dump the archive containing trained model weights
        n_way: number of classes per task
        n_shot: number of support examples per class
        n_query: number of query samples per class
        n_epochs: number of training epochs
        n_tasks_per_epoch: number of episodes per training epoch
        random_seed: defined random seed, for reproducibility
        device: what device to train the model on
        pretrained_weights: path to a tar archive of a pretrained model to start from
    """
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

    logger.info("Building model...")
    model = build_model(
        backbone=backbone,
        method=method,
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
    typer.run(main)

from math import floor

import numpy as np
from pathlib import Path

import click
import torch
import typer
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet

from src.constants import CIFAR_SPECS_DIR, TRAINED_MODELS_DIR, TB_LOGS_DIR, BACKBONES
from src.erm_trainer import ERMTrainer
from src.utils import set_random_seed


def main(
    backbone: str,
    specs_dir: Path = CIFAR_SPECS_DIR,
    output_model: Path = TRAINED_MODELS_DIR / "trained_classic.tar",
    n_epochs: int = 100,
    tb_log_dir: Path = TB_LOGS_DIR,
    random_seed: int = 0,
    device: str = "cuda",
):
    """

    Args:
        backbone: what model to train. Must be a key of constants.BACKBONES.
        specs_dir: where to find the dataset specs files
        output_model: where to dump the archive containing trained model weights
        n_epochs: number of training epochs
        tb_log_dir: where to dump tensorboard event files
        random_seed: defined random seed, for reproducibility
        device: what device to train the model on
    """
    n_workers = 12
    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()

    set_random_seed(random_seed)

    logger.info("Fetching training data...")
    whole_set = EasySet(specs_file=specs_dir / "train-val.json", training=True)
    train_set_size = floor(
        0.87 * len(whole_set)
    )  # Same factor as in the few-shot setting
    train_set, val_set = random_split(
        whole_set,
        [train_set_size, len(whole_set) - train_set_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )

    logger.info("Building model...")
    model = BACKBONES[backbone](pretrained=False).to(device)
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=len(set(train_set.labels)),
    ).to(device)

    logger.info("Starting training...")
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    trainer = ERMTrainer(
        optimizer=Adam(params=model.parameters()),
        loss_fn=loss_fn,
        device=device,
        tb_writer=SummaryWriter(log_dir=str(tb_log_dir)),
    )
    model = trainer.train(model, train_loader, n_epochs)

    model.fc = nn.Flatten()
    torch.save(model.state_dict(prefix="backbone."), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


if __name__ == "__main__":
    typer.run(main)

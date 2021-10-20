from pathlib import Path

import click
import torch
import typer
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet

from src.constants import CIFAR_SPECS_DIR, TRAINED_MODELS_DIR, TB_LOGS_DIR
from src.erm_trainer import ERMTrainer
from src.utils import set_random_seed


def main(
    specs_dir: Path = CIFAR_SPECS_DIR,
    output_model: Path = TRAINED_MODELS_DIR / "trained_classic.tar",
    n_epochs: int = 100,
    tb_log_dir: Path = TB_LOGS_DIR,
    random_seed: int = 0,
    device: str = "cuda",
):
    """

    Args:
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
    train_set = EasySet(specs_file=specs_dir / "train.json", training=True)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )

    logger.info("Building model...")
    model = resnet18(pretrained=False).to(device)
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=train_set.number_of_classes(),
    ).to(device)

    logger.info("Starting training...")
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    trainer = ERMTrainer(
        optimizer=Adam(params=model.parameters()),
        loss_fn=loss_fn,
        device=device,
        tb_writer=SummaryWriter(log_dir=tb_log_dir),
    )
    model = trainer.train(model, train_loader, n_epochs)

    model.fc = nn.Flatten()
    torch.save(model.state_dict(prefix="backbone."), output_model)
    logger.info(f"Pretrained model weights dumped at {output_model}")


if __name__ == "__main__":
    typer.run(main)

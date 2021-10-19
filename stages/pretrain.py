from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from easyfsl.data_tools import EasySet
from easyfsl.erm_trainer import ERMTrainer
from src.utils import set_random_seed


@click.option(
    "--n-epochs",
    help="Number of training epochs",
    type=int,
    default=100,
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
    n_epochs: int,
    specs_dir: Path,
    tb_log_dir: Path,
    output_model: Path,
    random_seed: int,
    device: str,
):
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
    main()

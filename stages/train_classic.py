from math import floor
from pathlib import Path
from statistics import mean

from easyfsl.data_tools import EasySet
from loguru import logger
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import typer
from tqdm import tqdm

from src.cifar import FewShotCIFAR100
from src.constants import CIFAR_SPECS_DIR, TRAINED_MODELS_DIR, TB_LOGS_DIR, BACKBONES
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

    set_random_seed(random_seed)

    logger.info("Fetching training data...")
    whole_set = FewShotCIFAR100(
        root=Path("data/cifar100/data"),
        specs_file=specs_dir / "train-val.json",
        training=True,
    )

    train_loader, val_loader = get_loaders(
        whole_set, batch_size, n_workers, random_seed
    )

    logger.info("Building model...")
    model = BACKBONES[backbone](
        pretrained=False, num_classes=len(set(whole_set.labels))
    ).to(device)
    model.device = device

    logger.info("Starting training...")
    model = train(model, n_epochs, train_loader, val_loader, tb_log_dir)

    torch.save(model.state_dict(prefix="backbone."), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


def get_loaders(whole_set, batch_size, n_workers, random_seed):
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

    return train_loader, val_loader


def train(model, n_epochs, train_loader, val_loader, tb_log_dir):
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    optimizer = Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model, average_loss = training_epoch(
            model, train_loader, optimizer, loss_fn, epoch
        )
        validation_accuracy = validate_model(model, val_loader)

        if tb_writer is not None:
            tb_writer.add_scalar("Train/loss", average_loss, epoch)
            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

    return model


def training_epoch(model, train_loader, optimizer, loss_fn, epoch):
    loss_list = []
    model.train()
    with tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
    ) as tqdm_train:
        for images, labels in tqdm_train:
            optimizer.zero_grad()
            loss = loss_fn(model(images.to(model.device)), labels.to(model.device))
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

            tqdm_train.set_postfix(loss=mean(loss_list))

    return model, mean(loss_list)


def validate_model(model, val_loader):
    model.eval()
    predictions_are_accurate = []
    for images, labels in val_loader:
        predictions_are_accurate += torch.max(model(images.to(model.device)), 1)[
            1
        ] == labels.to(model.device)
    average_accuracy = sum(predictions_are_accurate) / len(predictions_are_accurate)

    print(f"Validation accuracy: {(100 * average_accuracy):.2f}%")

    return average_accuracy


if __name__ == "__main__":
    typer.run(main)

from pathlib import Path
from statistics import mean

from loguru import logger
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import typer
from tqdm import tqdm

from src.constants import (
    TRAINED_MODELS_DIR,
    TB_LOGS_DIR,
    BACKBONES,
)
from src.few_shot_methods import SimpleShot
from src.utils.data_fetchers import (
    get_classic_loader,
    get_closed_task_loader,
)
from src.utils.utils import set_random_seed


def main(
    backbone: str,
    dataset: str,
    output_model: Path = TRAINED_MODELS_DIR / "trained_classic.tar",
    n_epochs: int = 200,
    scheduler_milestones: str = "160",
    scheduler_gamma: float = 0.1,
    batch_size: int = 512,
    learning_rate: float = 0.1,
    tb_log_dir: Path = TB_LOGS_DIR,
    random_seed: int = 0,
    device: str = "cuda",
):
    """

    Args:
        backbone: what model to train. Must be a key of constants.BACKBONES.
        dataset: what dataset to train the model on.
        output_model: where to dump the archive containing trained model weights
        n_epochs: number of training epochs
        scheduler_milestones: all milestones for optimizer scheduler, must be a string of
            comma-separated integers
        scheduler_gamma: discount factor for optimizer scheduler
        batch_size: the batch size
        learning_rate: optimizer's learning rate
        tb_log_dir: where to dump tensorboard event files
        random_seed: defined random seed, for reproducibility
        device: what device to train the model on
    """
    n_workers = 20

    set_random_seed(random_seed)

    logger.info("Fetching training data...")
    train_loader = get_classic_loader(
        dataset_name=dataset,
        split="train",
        training=True,
        batch_size=batch_size,
        n_workers=n_workers,
    )

    val_loader = get_closed_task_loader(
        dataset_name=dataset,
        n_way=5,
        n_shot=5,
        n_query=10,
        n_tasks=500,
        split="val",
        training=False,
        n_workers=n_workers,
    )

    logger.info("Building model...")
    model = BACKBONES[backbone](
        num_classes=len(set(train_loader.dataset.labels)), use_fc=True
    ).to(device)
    model.device = device

    logger.info("Starting training...")
    optimizer = SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    train_scheduler = MultiStepLR(
        optimizer,
        milestones=list(map(int, scheduler_milestones.split(","))),
        gamma=scheduler_gamma,
    )
    model = train(
        model=model,
        n_epochs=n_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=train_scheduler,
        tb_log_dir=tb_log_dir,
    )

    torch.save(model.state_dict(prefix="backbone."), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


def train(model, n_epochs, train_loader, val_loader, optimizer, scheduler, tb_log_dir):
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model, average_loss = training_epoch(
            model, train_loader, optimizer, loss_fn, epoch
        )
        validation_accuracy = validate_model(model, val_loader)

        if tb_writer is not None:
            tb_writer.add_scalar("Train/loss", average_loss, epoch)
            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

        scheduler.step(epoch)

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
    model.use_fc = False
    few_shot_model = SimpleShot(normalize_features=True)
    model.eval()
    predictions_are_accurate = []
    for _, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in val_loader:
        support_features = model(support_images)
        query_features = model(query_images)
        _, query_soft_predictions = few_shot_model(
            support_features, query_features, support_labels
        )
        predictions_are_accurate += torch.max(query_soft_predictions, 1)[
            1
        ] == query_labels.to(model.device)
    average_accuracy = sum(predictions_are_accurate) / len(predictions_are_accurate)

    print(f"Validation accuracy: {(100 * average_accuracy):.2f}%")

    model.use_fc = True

    return average_accuracy


if __name__ == "__main__":
    typer.run(main)

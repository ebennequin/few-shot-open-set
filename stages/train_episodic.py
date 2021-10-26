from pathlib import Path
from statistics import mean
from typing import Optional

from easyfsl.data_tools import EasySet, TaskSampler
from loguru import logger
import torch
from torch.optim import Adam, SGD
import typer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.cifar import FewShotCIFAR100
from src.constants import CIFAR_SPECS_DIR, TRAINED_MODELS_DIR, TB_LOGS_DIR
from src.utils import build_model, create_dataloader, set_random_seed


def main(
    backbone: str,
    feature_dimension: int = 256,
    method: str = "protonet",
    learning_rate: float = 0.01,
    specs_dir: Path = CIFAR_SPECS_DIR,
    output_model: Path = TRAINED_MODELS_DIR / "trained_episodic.tar",
    n_way: int = 5,
    n_shot: int = 5,
    n_query: int = 20,
    n_epochs: int = 200,
    n_tasks_per_epoch: int = 500,
    tb_log_dir: Path = TB_LOGS_DIR,
    random_seed: int = 0,
    device: str = "cuda",
    pretrained_weights: Optional[Path] = None,
):
    """
    Train a model in an episodic fashion.
    Args:
        backbone: what model to train. Must be a key of constants.BACKBONES.
        feature_dimension: dimension of the feature space
        method: what few-shot method to use during episodic training.
            Must be a key of constants.FEW_SHOT_METHODS.
        learning_rate: optimizer's learning rate
        specs_dir: where to find the dataset specs files
        output_model: where to dump the archive containing trained model weights
        n_way: number of classes per task
        n_shot: number of support examples per class
        n_query: number of query samples per class
        n_epochs: number of training epochs
        n_tasks_per_epoch: number of episodes per training epoch
        tb_log_dir: where to dump tensorboard event files
        random_seed: defined random seed, for reproducibility
        device: what device to train the model on
        pretrained_weights: path to a tar archive of a pretrained model to start from
    """
    n_validation_tasks = 100
    n_workers = 12

    set_random_seed(random_seed)

    logger.info("Fetching training data...")
    train_set = FewShotCIFAR100(
        root=Path("data/cifar100/data"),
        specs_file=specs_dir / "train_r.json",
        training=True,
    )
    train_sampler = TaskSampler(
        dataset=train_set,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks_per_epoch,
    )
    train_loader = create_dataloader(train_set, train_sampler, n_workers)

    logger.info("Fetching validation data...")
    val_set = FewShotCIFAR100(
        root=Path("data/cifar100/data"),
        specs_file=specs_dir / "val_r.json",
        training=False,
    )
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
        feature_dimension=feature_dimension,
        method=method,
        device=device,
        pretrained_weights=pretrained_weights,
    )

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.2)

    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    logger.info("Starting training...")
    for epoch in range(n_epochs):
        all_loss = []
        model.train()
        with tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
        ) as tqdm_train:
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_train:
                loss_value = model.fit_on_task(
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    optimizer,
                )
                all_loss.append(loss_value)

                tqdm_train.set_postfix(loss=mean(all_loss))

        validation_accuracy = model.validate(val_loader)

        if tb_writer is not None:
            tb_writer.add_scalar("Train/loss", mean(all_loss), epoch)
            tb_writer.add_scalar("Val/acc", validation_accuracy, epoch)

        train_scheduler.step(epoch)

    torch.save(model.state_dict(), output_model)
    logger.info(f"Trained model weights dumped at {output_model}")


if __name__ == "__main__":
    typer.run(main)

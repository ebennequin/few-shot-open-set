from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ERMTrainer:
    """
    Train a model with classical Empirical Risk Minimization.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        tb_writer: Optional[SummaryWriter],
    ):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.tb_writer = tb_writer

    def fit(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> (nn.Module, float):
        self.optimizer.zero_grad()
        scores = model(images)
        loss = self.loss_fn(scores, labels)
        loss.backward()
        self.optimizer.step()

        return model, loss.item()

    def training_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        epoch: int,
    ) -> (nn.Module, float):
        loss_list = []
        model.train()

        with tqdm(
            data_loader,
            desc=f"Epoch {epoch}",
        ) as tqdm_train:
            for images, labels in tqdm_train:
                model, loss_value = self.fit(
                    model, images.to(self.device), labels.to(self.device)
                )

                loss_list.append(loss_value)

                tqdm_train.set_postfix(loss=np.asarray(loss_list).mean())

        return model, np.asarray(loss_list).mean()

    def train(self, model, data_loader, n_epochs):
        for epoch in range(n_epochs):
            model, average_loss = self.training_epoch(model, data_loader, epoch)

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("PreTrain/loss", average_loss, epoch)

        return model

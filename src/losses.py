import argparse
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor


class _Loss(nn.Module):
    def __init__(
        self, args: argparse.Namespace, num_classes: int, reduction: str = "mean"
    ) -> None:
        super(_Loss, self).__init__()

        self.reduction: str = reduction
        assert 0 <= args.label_smoothing < 1
        self.label_smoothing: float = args.label_smoothing
        self.num_classes: int = num_classes

    def smooth_one_hot(self, targets: Tensor):
        with torch.no_grad():
            new_targets = torch.empty(
                size=(targets.size(0), self.num_classes), device=targets.device
            )
            new_targets.fill_(self.label_smoothing / (self.num_classes - 1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        return new_targets

    def loss_fn(self, logits: Tensor, one_hot_targets: Tensor):
        raise NotImplementedError


class _CrossEntropy(_Loss):
    def loss_fn(self, logits: Tensor, one_hot_targets: Tensor):
        logsoftmax_fn = nn.LogSoftmax(dim=1)
        logsoftmax = logsoftmax_fn(logits)
        loss = -(one_hot_targets * logsoftmax).sum(1)
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss

    def forward(self, logits: Tensor, targets: Tensor):
        one_hot_targets = self.smooth_one_hot(targets)
        return self.loss_fn(logits, one_hot_targets)

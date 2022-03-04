import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, List, Dict
from .feature_transforms import __dict__ as ALL_FEATURE_TRANSFORMS
from .aggregations import __dict__ as ALL_AGGREG


class AbstractFewShotMethod(nn.Module):
    """
    Abstract class for few-shot methods
    """

    def __init__(
        self,
        prepool_transforms: List[str],
        postpool_transforms: List[str],
        pool: bool,
        average_train_features: Tensor,
        std_train_features: Tensor,
        softmax_temperature: float = 1.0,
    ):
        super().__init__()
        self.softmax_temperature = softmax_temperature
        self.prepool_transforms = prepool_transforms
        self.postpool_transforms = postpool_transforms
        self.average_train_features = average_train_features
        self.std_train_features = std_train_features
        self.pool = pool
        self.prototypes: Tensor

    def forward(
        self, support_features: Tensor, query_features: Tensor, support_labels: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            support_features: support features
            query_features: query features
            support_labels: support labels

        Returns:
            support_soft_predictions: Tensor of shape [n_query, K], where K is the number of classes
                in the task, representing the soft predictions of the method for support samples.
            query_soft_predictions: Tensor of shape [n_query, K], where K is the number of classes
                in the task, representing the soft predictions of the method for query samples.
        """
        raise NotImplementedError

    def get_logits_from_euclidean_distances_to_prototypes(self, samples):
        return -self.softmax_temperature * torch.cdist(samples, self.prototypes)

    def get_logits_from_cosine_distances_to_prototypes(self, samples):
        return (
            self.softmax_temperature
            * F.normalize(samples, dim=1)
            @ F.normalize(self.prototypes, dim=1).T
        )

    @classmethod
    def from_cli_args(cls, args, average_train_features, std_train_features):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()},
            average_train_features=average_train_features,
            std_train_features=std_train_features
        )

from abc import abstractmethod
from typing import Tuple, List

import torch.nn.functional as F
from torch import Tensor, nn
import torch

EPSILON = 1e-10


class AbstractFeatureTransformer(nn.Module):
    @abstractmethod
    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "All feature transformers must implement a forward method."
        )


class IdentityTransformer(AbstractFeatureTransformer):
    """
    features: Tensor shape [N, hidden_dim, *]
    """

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return support_features, query_features


class Normalize(AbstractFeatureTransformer):
    """
    Inductive
    features: Tensor shape [N, hidden_dim, *]
    """

    def __init__(self, norm: float = 2.0):
        super().__init__()
        self.norm = norm

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return F.normalize(support_features, dim=1, p=self.norm), F.normalize(
            query_features, dim=1, p=self.norm
        )


class MaximumNormalize(AbstractFeatureTransformer):
    """
    Inductive
    Compute the maximum values accross support instances for each feature,
    and divide both support and query features by this tensor.
    features: Tensor shape [N, hidden_dim, *]
    """

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        norm_term = support_features.max(dim=0, keepdim=True).values
        return support_features / norm_term, query_features / norm_term


class LayerNorm(AbstractFeatureTransformer):
    """
    Inductive
    Normalize each instance in both support and query sets independently.
    Ensures that the features for an instance have mean 0 and variance 1.
    features: Tensor shape [N, hidden_dim, h, w]
    """

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        dims = (1, 2, 3)
        support_mean = torch.mean(support_features, dim=dims, keepdim=True)
        support_variance = torch.var(
            support_features, dim=dims, unbiased=False, keepdim=True
        )
        query_mean = torch.mean(query_features, dim=dims, keepdim=True)
        query_variance = torch.var(
            query_features, dim=dims, unbiased=False, keepdim=True
        )
        return (support_features - support_mean) / (
            support_variance.sqrt() + EPSILON
        ), (query_features - query_mean) / (query_variance.sqrt() + EPSILON)


class InductiveBatchNorm(AbstractFeatureTransformer):
    """
    Inductive
    From the support set, compute the mean and variance for each channel, and
    normalize both support and query instance wrt. this mean and variance.
    features: Tensor shape [N, hidden_dim, h, w]
    """

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        assert len(support_features.size()) >= 4
        dims = (0, 2, 3)
        mean = torch.mean(support_features, dim=dims, keepdim=True)
        var = torch.var(support_features, dim=dims, unbiased=False, keepdim=True)
        return (support_features - mean) / (var.sqrt() + EPSILON), (
            query_features - mean
        ) / (var.sqrt() + EPSILON)


class InstanceNorm(AbstractFeatureTransformer):
    """
    Inductive. Operate on the support and query sets independently.
    For each channel, instance, compute it's mean and variance across its width and height.
    Normalize the feature vector for this instance wrt. to this mean and variance.
    features: Tensor shape [N, hidden_dim, h, w]
    """

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        assert len(support_features.size()) >= 4
        dims = (2, 3)
        support_mean = torch.mean(support_features, dim=dims, keepdim=True)
        support_variance = torch.var(
            support_features, dim=dims, unbiased=False, keepdim=True
        )
        query_mean = torch.mean(query_features, dim=dims, keepdim=True)
        query_variance = torch.var(
            query_features, dim=dims, unbiased=False, keepdim=True
        )
        return (support_features - support_mean) / (
            support_variance.sqrt() + EPSILON
        ), (query_features - query_mean) / (query_variance.sqrt() + EPSILON)


class TransductiveBatchNorm(AbstractFeatureTransformer):
    """
    Transductive
    From all instances, compute the mean and variance for each channel, and
    normalize both support and query instance wrt. this mean and variance.
    features: Tensor shape [N, hidden_dim, h, w]
    """

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if len(support_features.size()) == 4:
            dims = (0, 2, 3)  # we normalize over the batch, as well as spatial dims
        elif len(support_features.size()) == 2:
            dims = (0,)
        else:
            raise ValueError("Problem with size of features.")
        all_features = torch.cat([support_features, query_features], 0)
        mean = torch.mean(all_features, dim=dims, keepdim=True)
        var = torch.var(all_features, dim=dims, unbiased=False, keepdim=True)
        return (support_features - mean) / (var.sqrt() + EPSILON), (
            query_features - mean
        ) / (var.sqrt() + EPSILON)


class PowerTransform(AbstractFeatureTransformer):
    """
    Inductive
    Elevate support and query features to power beta.
    features: Tensor shape [N, hidden_dim, *]
    """

    def __init__(self, beta: float = 0.5):
        super().__init__()
        self.beta = beta

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:

        return torch.pow(support_features + EPSILON, self.beta), torch.pow(
            query_features + EPSILON, self.beta
        )


class BaseSetCentering(AbstractFeatureTransformer):
    """
    Inductive
    Center support and query features on the average train features.
    features: Tensor shape [N, hidden_dim, *]
    """

    def __init__(self, average_train_features: Tensor):
        super().__init__()
        self.average_train_features = average_train_features

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if len(self.average_train_features.size()) != len(support_features.size()):
            self.average_train_features = self.average_train_features.squeeze(
                -1
            ).squeeze(-1)
        return (support_features - self.average_train_features), (
            query_features - self.average_train_features
        )


class SequentialFeatureTransformer(AbstractFeatureTransformer):
    """
    To apply several feature transformers sequentially.
    """

    def __init__(self, feature_transformers: List[AbstractFeatureTransformer]):
        super().__init__()
        self.feature_transformers = feature_transformers

    def forward(
        self, support_features: Tensor, query_features: Tensor
    ) -> Tuple[Tensor, Tensor]:
        for feature_transformer in self.feature_transformers:
            support_features, query_features = feature_transformer(
                support_features, query_features
            )

        return support_features, query_features


ALL_FEATURE_TRANSFORMERS = [
    IdentityTransformer,
    Normalize,
    MaximumNormalize,
    LayerNorm,
    InductiveBatchNorm,
    InstanceNorm,
    TransductiveBatchNorm,
    PowerTransform,
    BaseSetCentering,
]

from typing import Tuple

from torch import Tensor

from src.few_shot_methods import AbstractFewShotMethod
from easyfsl.utils import compute_prototypes


class SimpleShot(AbstractFewShotMethod):
    """
    Implementation of SimpleShot method https://arxiv.org/abs/1911.04623
    This is an inductive method.
    In this fashion, it comes down to Prototypical Networks.
    """

    def classify_support_and_queries(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:

        self.prototypes = compute_prototypes(support_features, support_labels)

        return (
            self.get_logits_from_euclidean_distances_to_prototypes(
                support_features
            ).softmax(-1),
            self.get_logits_from_euclidean_distances_to_prototypes(
                query_features
            ).softmax(-1),
        )

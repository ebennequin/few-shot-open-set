from typing import Tuple
from loguru import logger
from torch import Tensor

from .abstract import FewShotMethod
from easyfsl.utils import compute_prototypes


class SimpleShot(FewShotMethod):
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

        # print(support_features.size(), support_labels.size())
        self.prototypes = compute_prototypes(support_features, support_labels)

        probs_q = self.get_logits_from_cosine_distances_to_prototypes(query_features).softmax(-1)
        inliers = ~kwargs["outliers"].bool()
        acc = (probs_q.argmax(-1) == kwargs["query_labels"])[inliers].float().mean().item()
        return (
            self.get_logits_from_cosine_distances_to_prototypes(
                support_features
            ).softmax(-1),
            probs_q
        )

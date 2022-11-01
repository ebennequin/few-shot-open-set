from typing import Tuple
import torch.nn.functional as F
import torch
from torch import Tensor
from sklearn.metrics import precision_recall_curve
from .abstract import AllInOne
from easyfsl.utils import compute_prototypes
from sklearn.metrics import auc as auc_fn


class RobustEM(AllInOne):
    def __init__(
        self,
        inference_steps: int,
        lambda_s: float,
        lambda_z: float,
        ema_weight: float = 1.0,
    ):
        super().__init__()
        self.inference_steps = inference_steps
        self.lambda_s = lambda_s
        self.lambda_z = lambda_z
        self.ema_weight = ema_weight

    def cosine(self, X, Y):
        return F.normalize(X, dim=-1) @ F.normalize(Y, dim=-1).T

    def get_logits(self, prototypes, query_features):
        return self.cosine(query_features, prototypes)  # [query_size, num_classes]

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        # Metric dic
        num_classes = support_labels.unique().size(0)
        one_hot_labels = F.one_hot(
            support_labels, num_classes
        )  # [support_size, num_classes]
        support_size = support_features.size(0)

        prototypes = compute_prototypes(
            support_features, support_labels
        )  # [num_classes, feature_dim]
        soft_assignements = (1 / num_classes) * torch.ones(
            query_features.size(0), num_classes
        )  # [query_size, num_classes]
        inlier_scores = 0.5 * torch.ones((query_features.size(0), 1))

        acc_values = []
        auprs = []
        losses = []
        support_losses = []
        query_losses = []
        inlier_entropies = []
        soft_assignement_entropies = []
        inlier_scores_means = []
        inlier_scores_stds = []
        prototypes_norms = []

        for _ in range(self.inference_steps):

            # Compute inlier scores
            logits_q = self.get_logits(
                prototypes, query_features
            )  # [query_size, num_classes]
            inlier_scores = (
                self.ema_weight
                * (
                    (soft_assignements * logits_q / self.lambda_s)
                    .sum(-1, keepdim=True)
                    .sigmoid()
                )
                + (1 - self.ema_weight) * inlier_scores
            )  # [query_size, 1]

            # Compute new assignements
            soft_assignements = (
                self.ema_weight
                * ((inlier_scores * logits_q / self.lambda_z).softmax(-1))
                + (1 - self.ema_weight) * soft_assignements
            )  # [query_size, num_classes]

            # COmpute metrics
            outliers = kwargs["outliers"].bool()
            outlier_scores = 1 - inlier_scores
            inliers = ~outliers
            acc = (
                (soft_assignements.argmax(-1) == kwargs["query_labels"])[inliers]
                .float()
                .mean()
                .item()
            )
            acc_values.append(acc)
            precision, recall, thresholds = precision_recall_curve(
                outliers.numpy(), outlier_scores.numpy()
            )
            aupr = auc_fn(recall, precision)
            auprs.append(aupr)
            precision, recall, thresholds = precision_recall_curve(
                (~inliers).numpy(), outlier_scores.numpy()
            )

            support_loss = soft_cross_entropy(
                self.get_logits(prototypes, support_features), one_hot_labels
            )
            query_loss = soft_cross_entropy(
                self.get_logits(prototypes, query_features),
                soft_assignements,
                inlier_scores,
            )
            inlier_entropy = binary_entropy(inlier_scores)
            soft_assignement_entropy = entropy(soft_assignements)
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            inlier_entropies.append(inlier_entropy)
            soft_assignement_entropies.append(soft_assignement_entropy)
            losses.append(
                support_loss
                + query_loss
                + self.lambda_s * inlier_entropy
                + self.lambda_z * soft_assignement_entropy
            )
            inlier_scores_means.append(inlier_scores.mean())
            inlier_scores_stds.append(inlier_scores.std())
            prototypes_norms.append(prototypes.norm(dim=-1).mean())

            # Compute new prototypes
            all_features = torch.cat(
                [support_features, query_features], 0
            )  # [support_size + query_size, feature_dim]
            all_assignements = torch.cat(
                [one_hot_labels, soft_assignements], dim=0
            )  # [support_size + query_size, num_classes]
            all_inliers_scores = torch.cat(
                [torch.ones(support_size, 1), inlier_scores], 0
            )  # [support_size + query_size, 1]
            # TODO : scaler par S/Q au numérateur et au dénominateur
            prototypes = (
                self.ema_weight
                * (
                    (all_inliers_scores * all_assignements).T
                    @ all_features
                    / (all_inliers_scores * all_assignements).sum(0).unsqueeze(1)
                )
                + (1 - self.ema_weight) * prototypes
            )  # [num_classes, feature_dim]

        kwargs["intra_task_metrics"]["main_metrics"]["acc"].append(acc_values)
        kwargs["intra_task_metrics"]["main_metrics"]["aupr"].append(auprs)
        kwargs["intra_task_metrics"]["secondary_metrics"]["losses"].append(losses)
        kwargs["intra_task_metrics"]["secondary_metrics"]["support_losses"].append(
            support_losses
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["query_losses"].append(
            query_losses
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_entropies"].append(
            inlier_entropies
        )
        kwargs["intra_task_metrics"]["secondary_metrics"][
            "soft_assignement_entropies"
        ].append(soft_assignement_entropies)
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_scores_means"].append(
            inlier_scores_means
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["inlier_scores_stds"].append(
            inlier_scores_stds
        )
        kwargs["intra_task_metrics"]["secondary_metrics"]["prototypes_norms"].append(
            prototypes_norms
        )
        return (
            self.get_logits(prototypes, support_features).softmax(-1),
            self.get_logits(prototypes, query_features).softmax(-1),
            outlier_scores,
        )


def soft_cross_entropy(logits, soft_labels, _inlier_scores=None):
    _inlier_scores = (
        _inlier_scores if _inlier_scores is not None else torch.ones(len(logits))
    )
    return -((logits * soft_labels).sum(dim=1) * _inlier_scores).mean()


def binary_entropy(scores):
    return -(scores * scores.log() + (1 - scores) * (1 - scores).log()).mean()


def entropy(scores):
    return -(scores * scores.log()).sum(dim=1).mean()

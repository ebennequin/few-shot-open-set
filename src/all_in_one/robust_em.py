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
        softmax_temperature: float,
        inference_steps: int,
        lambda_s: float,
        lambda_z: float,
    ):
        super().__init__()
        self.inference_steps = inference_steps
        self.softmax_temperature = softmax_temperature
        self.lambda_s = lambda_s
        self.lambda_z = lambda_z

    def cosine(self, X, Y):
        return F.normalize(X, dim=-1) @ F.normalize(Y, dim=-1).T

    def get_logits(self, prototypes, query_features):
        return self.softmax_temperature * self.cosine(
            query_features, prototypes
        )  # [query_size, num_classes]

    def __call__(
        self,
        support_features: Tensor,
        query_features: Tensor,
        support_labels: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        self.iter_ = 0

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
        inlier_scores = 0.5 * torch.ones(query_features.size(0))

        acc_values = []
        auprs = []

        for self.iter_ in range(self.inference_steps):

            # Compute inlier scores
            logits_q = self.get_logits(
                prototypes, query_features
            )  # [query_size, num_classes]
            inlier_scores = (
                (soft_assignements * logits_q / self.lambda_s)
                .sum(-1, keepdim=True)
                .sigmoid()
            )  # [query_size, 1]

            # Compute new assignements
            soft_assignements = (inlier_scores * logits_q / self.lambda_z).softmax(
                -1
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
                (all_inliers_scores * all_assignements).T
                @ all_features
                / (all_inliers_scores * all_assignements).sum(0).unsqueeze(1)
            )  # [num_classes, feature_dim]

        kwargs["intra_task_metrics"]["main_metrics"]["acc"].append(acc_values)
        kwargs["intra_task_metrics"]["main_metrics"]["aupr"].append(auprs)
        return (
            self.get_logits(prototypes, support_features).softmax(-1),
            self.get_logits(prototypes, query_features).softmax(-1),
            outlier_scores,
        )

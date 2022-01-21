"""
Functions used to compute outlier scores from classification predictions.
"""
import pandas as pd
import torch
from tqdm import tqdm


def get_pseudo_renyi_entropy(soft_predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute the pseudo-Renyi entropy of the prediction for each query, i.e. the sum of the
        squares of each class' classification score.
    Args:
        soft_predictions: predictions before softmax, shape (n_query*n_way, feature_dimension)
    Returns:
        1-dim tensor of length (n_query*n_way) giving the prediction entropy for each query
    """
    return torch.pow(soft_predictions, 2).sum(dim=1).detach().cpu()


def get_shannon_entropy(soft_predictions: torch.Tensor) -> torch.Tensor:
    """
    Compute the Shannon entropy of the prediction for each query.
    Args:
        soft_predictions: predicted probabilities, shape (n_query*n_way, feature_dimension)
    Returns:
        1-dim tensor of length (n_query*n_way) giving the prediction entropy for each query
    """
    return (soft_predictions * torch.log(soft_predictions)).sum(dim=1).detach().cpu()


def compute_outlier_scores_with_renyi_divergence(
    soft_predictions: torch.Tensor,
    soft_support_predictions: torch.Tensor,
    alpha: int = 2,
    method: str = "topk",
    k: int = 3,
) -> torch.Tensor:
    """
    Compute all Renyi divergences from query instances to support instances, and assign an outlier
    score to each query from its divergence with it's "neighbouring" supports w.r.t. to this
    divergence.
    Args:
        soft_predictions: predictions, shape (n_query*n_way, feature_dimension)
        soft_support_predictions: predictions for support examples, shape (n_shot*n_way, feature_dimension)
        alpha: parameter alpha for Renyi divergence
        method: min or topk. min returns for each query its divergence with the "nearest" support
            example. topk returns for each query with divergence with the k-th "nearest" support
            example.
        k: only used if method=topk. Defines the k.

    Returns:
        1-dim tensor of length (n_query*n_way) containing the outlier score of each query
    """
    pairwise_divergences = (1 / (alpha - 1)) * torch.log(
        torch.pow(soft_predictions, alpha).matmul(
            torch.pow(soft_support_predictions, 1 - alpha).T
        )
    )

    if method == "min":
        return 1 - pairwise_divergences.min(dim=1)[0]
    elif method == "topk":
        return 1 - pairwise_divergences.topk(k, dim=1, largest=False)[0].mean(-1)
    else:
        raise ValueError("Don't know this method.")


def detect_outliers(outlier_detector, data_loader, n_way, n_query):
    outlier_detection_df_list = []
    for task_id, (
        support_features,
        support_labels,
        query_features,
        query_labels,
        _,
    ) in tqdm(enumerate(data_loader)):
        outlier_detection_df_list.append(
            pd.DataFrame(
                {
                    "task": task_id,
                    "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                    "outlier_score": outlier_detector(
                        support_features, support_labels, query_features, query_labels
                    ),
                }
            )
        )

    return pd.concat(outlier_detection_df_list, ignore_index=True)

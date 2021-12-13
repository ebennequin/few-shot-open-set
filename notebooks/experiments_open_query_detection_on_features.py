"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

#%%
import pickle
from pathlib import Path
from statistics import mean

import argparse
import pandas as pd
import torch
from pyod.models.knn import KNN
from tqdm import tqdm

from src.datasets import FeaturesDataset
from src.utils.utils import (
    set_random_seed,
)
from src.few_shot_methods import __dict__ as all_methods
from src.utils.outlier_detectors import (
    compute_outlier_scores_with_renyi_divergence,
)
from src.utils.plots_and_metrics import show_all_metrics_and_plots
from src.utils.data_fetchers import create_dataloader
from src.open_query_sampler import OpenQuerySampler

#%% Constants

n_way: int = 10
n_shot: int = 500
n_query: int = 10
n_tasks: int = 500
random_seed: int = 0
n_workers = 12

set_random_seed(random_seed)


def parse_args() -> argparse.Namespace:

    # Data
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_choice', type=str, default='cifar')
    parser.add_argument('--n_way', type=int, default=10)
    parser.add_argument('--n_shot', type=int, default=500)
    parser.add_argument('--n_query', type=int, default=10)
    parser.add_argument('--n_taks', type=int, default=500)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--n_workers', type=int, default=12)

    # Model
    parser.add_argument('--backbone_choice', type=str, default='resnet18')
    parser.add_argument('--training', type=str, default='classic')

    # Method
    parser.add_argument('--inference_method', type=str, default='SimpleShot')
    parser.add_argument('--softmax_temperature', type=float, default=1.0)
    parser.add_argument('--inference_lr', type=float, default=1e-3,
                        help='Learning rate used for methods that perform \
                        gradient-based inference.')
    parser.add_argument('--inference_steps', type=float, default=10,
                        help='Steps used for gradient-based inferences.')

    args = parser.parse_args()
    return args


def main(args):
    #%% Load pre-computed features for the selected dataset.

    pickle_basename = f"{args.backbone_choice}_{args.dataset_choice}_{args.training}.pickle"
    features_path = Path("data/features") / args.dataset_choice / "test" / pickle_basename
    train_features_path = Path("data/features") / args.dataset_choice / "train" / pickle_basename

    with open(features_path, "rb") as stream:
        features = pickle.load(stream)

    # We also load features from the train set to center query features on the average train set
    # feature vector
    with open(train_features_path, "rb") as stream:
        train_features = pickle.load(stream)
        average_train_features = torch.cat(
            [
                torch.from_numpy(features_per_label)
                for features_per_label in train_features.values()
            ],
            dim=0,
        ).mean(0)

    #%% Create data loader

    dataset = FeaturesDataset(features, features_to_center_on=average_train_features)
    sampler = OpenQuerySampler(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    data_loader = create_dataloader(dataset=dataset, sampler=sampler, n_workers=n_workers)


    #%%% Create method
    method = all_methods[args.inference_method](args)

    #%% Outlier detection strategies using model's predictions

    accuracy_list = []
    outlier_detection_df_list = []
    for support_features, support_labels, query_features, query_labels, _ in tqdm(
        data_loader
    ):
        # support_prototypes = compute_prototypes(support_features, support_labels)  # [K, d]

        # predictions = -torch.cdist(query_features, support_prototypes)  # [n_queries, K]
        soft_preds_s, soft_preds_q = method(feat_s=support_features, y_s=support_labels, feat_q=query_features)
        hard_preds_q = soft_preds_q.argmax(-1)

        # Accuracies (to get an evaluation of the model along with the ROC)
        accuracy_list.append(
            (
                hard_preds_q[: n_way * n_query].detach().data
                == query_labels[: n_way * n_query]
            )
            .sum()
            .item()
            / (n_way * n_query)
        )

        # TODO: play with softmax's temperature
        # Build outlier detection dataframe
        outlier_detection_df_list.append(
            pd.DataFrame(
                {
                    "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                    "outlier_score": compute_outlier_scores_with_renyi_divergence(
                        soft_predictions=soft_preds_q,
                        soft_support_predictions=soft_preds_s,
                        alpha=-5,
                    )
                        # -torch.cdist(support_features, support_prototypes),
                    # "outlier_score": get_pseudo_renyi_entropy(predictions),
                    # "outlier_score": get_shannon_entropy(predictions),
                }
            )
        )

    outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

    print(f"Average accuracy: {(100 * mean(accuracy_list)):.2f}%")
    show_all_metrics_and_plots(
        outlier_detection_df, title="Outlier Detection on Classification Scores"
    )

    #%% Outlier detection strategies using feature vectors

    outlier_detection_df_list = []
    for support_features, support_labels, query_features, query_labels, _ in tqdm(
        data_loader
    ):
        # TODO: LOF gives lower score to outliers and KNN and IFOREST give higher score to outliers
        # TODO: behaviour described in pyod's doc says it should give higher score to outliers
        # clustering = LocalOutlierFactor(n_neighbors=3, novelty=True, metric="euclidean")
        # clustering = IForest(n_estimators=100, n_jobs=-1)
        clustering = KNN(n_neighbors=3, method="mean", n_jobs=-1)
        clustering.fit(support_features)

        outlier_detection_df_list.append(
            pd.DataFrame(
                {
                    "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                    "outlier_score": 1 - clustering.decision_function(query_features),
                }
            )
        )

    outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

    show_all_metrics_and_plots(outlier_detection_df, title="Outlier Detection on Features")


if __name__ == "__main__":
    args = parse_args()
    main(args)

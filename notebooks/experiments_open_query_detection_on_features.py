"""
Load the features extracted from a dataset's images, sample Open Set Few-Shot Classification Tasks
and infer various outlier detection methods en them.
"""

#%%
import pickle
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
from easyfsl.utils import compute_prototypes
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from src.datasets import FeaturesDataset
from src.utils.utils import (
    set_random_seed,
)
from src.utils.outlier_detectors import (
    compute_outlier_scores_with_renyi_divergence,
    get_shannon_entropy,
    get_pseudo_renyi_entropy,
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

DATASET_CHOICE = "cifar"
# DATASET_CHOICE = "mini_imagenet"
BACKBONE_CHOICE = "resnet18"
# BACKBONE_CHOICE = "resnet12"
TRAINING_METHOD_CHOICE = "classic"
# TRAINING_METHOD_CHOICE = "episodic"

#%% Load pre-computed features for the selected dataset.

pickle_basename = f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_{TRAINING_METHOD_CHOICE}.pickle"
features_path = Path("data/features") / DATASET_CHOICE / "test" / pickle_basename
train_features_path = Path("data/features") / DATASET_CHOICE / "train" / pickle_basename

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


#%% Outlier detection strategies using model's predictions

accuracy_list = []
outlier_detection_df_list = []
for support_features, support_labels, query_features, query_labels, _ in tqdm(
    data_loader
):
    support_prototypes = compute_prototypes(support_features, support_labels)

    predictions = -torch.cdist(query_features, support_prototypes)

    # Accuracies (to get an evaluation of the model along with the ROC)
    accuracy_list.append(
        (
            torch.max(
                predictions[: n_way * n_query].detach().data,
                1,
            )[1]
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
                    predictions,
                    -torch.cdist(support_features, support_prototypes),
                    alpha=-5,
                )
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

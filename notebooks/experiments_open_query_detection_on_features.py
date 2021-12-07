#%%
import pickle
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
from easyfsl.utils import compute_prototypes
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from src.datasets import FeaturesDataset
from src.utils import (
    set_random_seed,
    get_pseudo_renyi_entropy,
    create_dataloader,
    show_all_metrics_and_plots,
)
from src.open_query_sampler import OpenQuerySampler

#%%

n_way: int = 5
n_shot: int = 5
n_query: int = 10
n_tasks: int = 500
random_seed: int = 0
n_workers = 12

set_random_seed(random_seed)

# DATASET_CHOICE = "cifar"
DATASET_CHOICE = "mini_imagenet"
BACKBONE_CHOICE = "resnet18"
TRAINING_METHOD_CHOICE = "classic"

pickle_basename = f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_{TRAINING_METHOD_CHOICE}.pickle"
features_path = Path("data/features") / DATASET_CHOICE / "test" / pickle_basename
train_features_path = Path("data/features") / DATASET_CHOICE / "train" / pickle_basename

#%%
with open(features_path, "rb") as stream:
    features = pickle.load(stream)

with open(train_features_path, "rb") as stream:
    train_features = pickle.load(stream)
    average_train_features = torch.cat(
        [
            torch.from_numpy(features_per_label)
            for features_per_label in train_features.values()
        ],
        dim=0,
    ).mean(0)


#%%
# dataset = FeaturesDataset(features, features_to_center_on=average_train_features)
dataset = FeaturesDataset(features)
sampler = OpenQuerySampler(
    dataset=dataset,
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_tasks=n_tasks,
)
data_loader = create_dataloader(dataset=dataset, sampler=sampler, n_workers=n_workers)


#%% Test DOCTOR strategy

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

    # Build outlier detection dataframe
    outlier_detection_df_list.append(
        pd.DataFrame(
            {
                "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                "outlier_score": get_pseudo_renyi_entropy(predictions),
                # "outlier_score": get_shannon_entropy(predictions),
            }
        )
    )

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

print(f"Average accuracy: {(100 * mean(accuracy_list)):.2f}%")
show_all_metrics_and_plots(outlier_detection_df, title="DOCTOR")


#%% Test LocalOutlierFactor

outlier_detection_df_list = []
for support_features, support_labels, query_features, query_labels, _ in tqdm(
    data_loader
):
    clustering = LocalOutlierFactor(n_neighbors=3, novelty=True, metric="euclidean")
    # clustering = IsolationForest()
    clustering.fit(support_features)

    outlier_detection_df_list.append(
        pd.DataFrame(
            {
                "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                "outlier_score": clustering.decision_function(query_features),
            }
        )
    )

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

show_all_metrics_and_plots(outlier_detection_df, title="LOF")

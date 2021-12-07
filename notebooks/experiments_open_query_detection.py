#%%
from pathlib import Path
from statistics import mean

import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from easyfsl.methods import PrototypicalNetworks
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import (
    BACKBONES,
)

from src.utils import (
    get_classic_loader,
    get_inference_model,
    get_pseudo_renyi_entropy,
    get_task_loader,
    plot_episode,
    plot_roc,
    plot_twin_hist,
    set_random_seed,
    show_all_metrics_and_plots,
)

#%%

n_way: int = 5
n_shot: int = 5
n_query: int = 10
n_tasks: int = 50
random_seed: int = 0
device: str = "cuda:2"  # Change to 'cuda' if you have NVidia GPU
n_workers = 12

set_random_seed(random_seed)

DATASET_CHOICE = "cifar"
# DATASET_CHOICE = "mini_imagenet"
BACKBONE_CHOICE = "resnet18"

# model_weights = Path("data/models") / f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_episodic.tar"
model_weights = Path("data/models") / f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_classic.tar"

#%%
data_loader = get_task_loader(DATASET_CHOICE, n_way, n_shot, n_query, n_tasks)

#%%
# TODO: tester
model = get_inference_model(
    backbone=BACKBONES[BACKBONE_CHOICE](),
    weights_path=model_weights,
    train_loader=get_classic_loader(
        DATASET_CHOICE,
    ),
)

#%%
one_episode = next(iter(data_loader))
plot_episode(one_episode[0], one_episode[2])

#%% Test DOCTOR strategy

accuracy_list = []
outlier_detection_df_list = []
model.eval()
with torch.no_grad():
    for support_images, support_labels, query_images, query_labels, _ in tqdm(
        data_loader
    ):
        model.process_support_set(support_images.to(device), support_labels.to(device))
        predictions = model(query_images.to(device))

        # Accuracies (to get an evaluation of the model along with the ROC)
        accuracy_list.append(
            (
                torch.max(
                    predictions[: n_way * n_query].detach().data,
                    1,
                )[1]
                == query_labels[: n_way * n_query].to(device)
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
# Change to pyod LocalOutlierFactor and integrate into next cell
# See issues here: https://pyod.readthedocs.io/en/latest/issues.html

outlier_detection_df_list = []
for support_images, support_labels, query_images, query_labels, _ in tqdm(data_loader):
    support_features = nn.functional.normalize(
        model.backbone(support_images.to(device)), dim=1
    )
    query_features = nn.functional.normalize(
        model.backbone(query_images.to(device)), dim=1
    )

    clustering = LocalOutlierFactor(n_neighbors=3, novelty=True, metric="euclidean")
    clustering.fit(support_features.detach().cpu())

    outlier_detection_df_list.append(
        pd.DataFrame(
            {
                "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                "outlier_score": clustering.decision_function(
                    query_features.detach().cpu()
                ),
            }
        )
    )

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

show_all_metrics_and_plots(outlier_detection_df, title="LOF")

#%% Test IForest/KNN
# Ajouter la gestion du score pyod en paramètre

outlier_detection_df_list = []
for support_images, support_labels, query_images, query_labels, _ in tqdm(data_loader):
    support_features = nn.functional.normalize(
        model.backbone(support_images.to(device)), dim=1
    )
    query_features = nn.functional.normalize(
        model.backbone(query_images.to(device)), dim=1
    )

    # clustering = IForest(n_estimators=100, n_jobs=-1)
    clustering = KNN(n_neighbors=3, method="largest", n_jobs=-1)
    clustering.fit(support_features.detach().cpu())

    outlier_detection_df_list.append(
        pd.DataFrame(
            {
                "outlier": (n_way * n_query) * [False] + (n_way * n_query) * [True],
                "outlier_score": 1.0
                - clustering.decision_function(query_features.detach().cpu()),
            }
        )
    )

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

show_all_metrics_and_plots(outlier_detection_df, title="LOF")

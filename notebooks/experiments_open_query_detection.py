#%%
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from src.constants import (
    BACKBONES,
)

from src.utils import (
    set_random_seed,
    plot_episode,
    plot_roc,
    plot_twin_hist,
    get_pseudo_renyi_entropy,
    get_task_loader, get_classic_loader, get_inference_model,
)

#%%

n_way: int = 5
n_shot: int = 5
n_query: int = 10
n_tasks: int = 500
random_seed: int = 0
device: str = "cuda"
n_workers = 12

set_random_seed(random_seed)

DATASET_CHOICE = "cifar"
# DATASET_CHOICE = "mini_imagenet"
BACKBONE_CHOICE = "resnet18"

# model_weights = Path("data/models") / f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_episodic.tar"
model_weights = Path("data/models") / f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_classic.tar"

#%%


#%%
data_loader = get_task_loader(DATASET_CHOICE, n_way, n_shot, n_query, n_tasks)

# TODO: tester 
model = get_inference_model(
    BACKBONES[BACKBONE_CHOICE](),
    model_weights,
    get_classic_loader(
        DATASET_CHOICE,
    ),
)

#%%
one_episode = next(iter(data_loader))
plot_episode(one_episode[0], one_episode[2])

#%%


def show_all_metrics_and_plots(outliers_df, title, objective=0.9):
    roc_auc = plot_roc(outliers_df, title=title)
    print(f"ROC AUC: {roc_auc}")

    precisions, recalls, _ = precision_recall_curve(
        outliers_df.outlier, -outliers_df.outlier_score
    )
    precision_at_recall_objective = precisions[
        next(i for i, value in enumerate(recalls) if value < objective)
    ]
    recall_at_precision_objective = recalls[
        next(i for i, value in enumerate(precisions) if value > objective)
    ]
    print(f"Precision for recall={objective}: {precision_at_recall_objective}")
    print(f"Recall for precision={objective}: {recall_at_precision_objective}")

    plot_twin_hist(outlier_detection_df, title=title)


#%% Test DOCTOR strategy

accuracy_list = []
outlier_detection_df_list = []
model.eval()
with torch.no_grad():
    for support_images, support_labels, query_images, query_labels, _ in tqdm(
        data_loader
    ):
        model.process_support_set(support_images.cuda(), support_labels.cuda())
        predictions = model(query_images.cuda())

        # Accuracies (to get an evaluation of the model along with the ROC)
        accuracy_list.append(
            (
                torch.max(
                    predictions[: n_way * n_query].detach().data,
                    1,
                )[1]
                == query_labels[: n_way * n_query].cuda()
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
for support_images, support_labels, query_images, query_labels, _ in tqdm(data_loader):
    support_features = model.backbone(support_images.cuda())
    query_features = model.backbone(query_images.cuda())

    clustering = LocalOutlierFactor(n_neighbors=3, novelty=True, metric="euclidean")
    # clustering = IsolationForest()
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

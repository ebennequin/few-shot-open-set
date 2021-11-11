#%%
from pathlib import Path
from statistics import mean

from easyfsl.methods import PrototypicalNetworks
from easyfsl.utils import compute_prototypes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from src.cifar import FewShotCIFAR100
from src.constants import (
    MINI_IMAGENET_ROOT_DIR,
    MINI_IMAGENET_SPECS_DIR,
    CIFAR_SPECS_DIR,
    CIFAR_ROOT_DIR,
)
from src.mini_imagenet import MiniImageNet
from src.open_query_sampler import OpenQuerySampler

from src.utils import set_random_seed, create_dataloader, build_model, plot_episode

#%%

n_way: int = 5
n_shot: int = 5
n_query: int = 10
n_epochs: int = 200
n_tasks_per_epoch: int = 500
random_seed: int = 0
device: str = "cuda"
n_validation_tasks = 100
n_workers = 12


#%%

set_random_seed(random_seed)

# dataset = FewShotCIFAR100(
#     root=CIFAR_ROOT_DIR,
#     specs_file=CIFAR_SPECS_DIR / "test.json",
#     training=False,
# )
dataset = MiniImageNet(
    root=MINI_IMAGENET_ROOT_DIR,
    specs_file=MINI_IMAGENET_SPECS_DIR / "test_images.csv",
    training=False,
)
sampler = OpenQuerySampler(
    dataset=dataset,
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_tasks=n_tasks_per_epoch,
)
data_loader = create_dataloader(dataset, sampler, n_workers)

#%%
one_episode = next(iter(data_loader))
plot_episode(one_episode[0], one_episode[2])

#%%
backbone = resnet18(num_classes=512)
model = PrototypicalNetworks(backbone).cuda()
# model.load_state_dict(torch.load("data/models/resnet18_cifar_episodic.tar"))
model.load_state_dict(torch.load("data/models/resnet18_mini_imagenet_episodic.tar"))
# model.backbone.fc = nn.Flatten()
model.eval()

#%%
def get_pseudo_renyi_entropy(predictions):
    return (
        torch.pow(nn.functional.softmax(predictions, dim=1), 2)
        .sum(dim=1)
        .detach()
        .cpu()
    )


def plot_roc(outliers_df, title):
    gamma_range = np.linspace(0.0, 1.0, 1000)
    precisions = []
    recall = []

    for gamma in gamma_range:
        this_gamma_detection_df = outliers_df.assign(
            outlier_prediction=lambda df: df.outlier_score < gamma
        )
        precisions.append(
            (
                this_gamma_detection_df.outlier
                & this_gamma_detection_df.outlier_prediction
            ).sum()
            / (this_gamma_detection_df.outlier.sum() + 1)
        )
        recall.append(
            (
                ~this_gamma_detection_df.outlier
                & this_gamma_detection_df.outlier_prediction
            ).sum()
            / ((~this_gamma_detection_df.outlier).sum() + 1)
        )

    plt.plot(recall, precisions)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.show()


#%%
# Compute the mean of all features extracted from the train set
# To follow Bertinetto's inference strategy
def get_train_set_feature_mean(model):
    # train_set = FewShotCIFAR100(
    #     root=CIFAR_ROOT_DIR,
    #     specs_file=CIFAR_SPECS_DIR / "train.json",
    #     training=False,
    # )
    train_set = MiniImageNet(
        root=MINI_IMAGENET_ROOT_DIR,
        specs_file=MINI_IMAGENET_SPECS_DIR / "train_images.csv",
        training=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=1024,
        num_workers=n_workers,
        pin_memory=True,
    )

    model.eval()
    with torch.no_grad():
        all_features = []
        for images, _ in tqdm(train_loader):
            all_features.append(model.backbone(images.cuda()).data)

    return torch.cat(all_features, dim=0).mean(0)


train_features_mean = get_train_set_feature_mean(model)


#%% Test DOCTOR strategy

accuracy_list = []
outlier_detection_df_list = []
model.eval()
with torch.no_grad():
    for support_images, support_labels, query_images, query_labels, _ in tqdm(
        data_loader
    ):
        # model.process_support_set(support_images.cuda(), support_labels.cuda())
        # predictions = model(query_images.cuda())

        # Follow the inference strategy by Bertinetto
        # (found that it improves the ROC curve and a little bit the accuracy (~+1%))
        support_features = nn.functional.normalize(
            model.backbone(support_images.cuda()).data - train_features_mean, dim=1
        )
        model.prototypes = compute_prototypes(support_features, support_labels)
        query_features = nn.functional.normalize(
            model.backbone(query_images.cuda()).data - train_features_mean, dim=1
        )
        predictions = -torch.cdist(query_features, model.prototypes)

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
                }
            )
        )

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

print(f"Average accuracy: {(100 * mean(accuracy_list)):.2f}%")
plot_roc(outlier_detection_df, title="DOCTOR")


#%% Test LocalOutlierFactor

outlier_detection_df_list = []
for support_images, support_labels, query_images, query_labels, _ in tqdm(data_loader):
    support_features = model.backbone(support_images.cuda())
    query_features = model.backbone(query_images.cuda())

    clustering = LocalOutlierFactor(
        n_neighbors=3, novelty=True, metric="euclidean"
    ).fit(support_features.detach().cpu())

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

plot_roc(outlier_detection_df, title="Local Outlier Factor")

# TODO: tester LocalOutlierFactor item par item pour voir si c'est différent

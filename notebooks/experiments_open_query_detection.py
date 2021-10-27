#%%
from pathlib import Path

from easyfsl.methods import PrototypicalNetworks
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm

from src.cifar import FewShotCIFAR100
from src.open_query_sampler import OpenQuerySampler

from src.utils import set_random_seed, create_dataloader, build_model, plot_episode

#%%

cifar_root = Path("data") / "cifar100"
data_root = cifar_root / "data"
specs_dir = cifar_root / "specs"
n_way: int = 5
n_shot: int = 5
n_query: int = 5
n_epochs: int = 200
n_tasks_per_epoch: int = 500
random_seed: int = 0
device: str = "cuda"
n_validation_tasks = 100
n_workers = 12


#%%

set_random_seed(random_seed)

train_set = FewShotCIFAR100(
    root=cifar_root / "data",
    specs_file=specs_dir / "train.json",
    training=False,
)
train_sampler = OpenQuerySampler(
    dataset=train_set,
    n_way=n_way,
    n_shot=n_shot,
    n_query=n_query,
    n_tasks=n_tasks_per_epoch,
)
train_loader = create_dataloader(train_set, train_sampler, n_workers)

#%%
one_episode = next(iter(train_loader))
plot_episode(one_episode[0], one_episode[2])

#%%
backbone = resnet18(num_classes=256)
model = PrototypicalNetworks(backbone).cuda()
model.load_state_dict(torch.load("data/models/resnet18_episodic.tar"))
# model.backbone.fc = nn.Flatten()

#%%
def get_pseudo_renyi_entropy(predictions):
    return torch.pow(nn.functional.softmax(predictions, dim=1), 2).sum(dim=1).detach().cpu()


def plot_roc(outliers_df):
    gamma_range = np.linspace(0.,1.,1000)
    precisions = []
    recall = []

    for gamma in gamma_range:
        this_gamma_detection_df = outliers_df.assign(
            outlier_prediction=lambda df: df.outlier_score < gamma
        )
        precisions.append(
            (this_gamma_detection_df.outlier & this_gamma_detection_df.outlier_prediction).sum()
            / (this_gamma_detection_df.outlier.sum() + 1)
        )
        recall.append(
            (~this_gamma_detection_df.outlier & this_gamma_detection_df.outlier_prediction).sum()
            / ((~this_gamma_detection_df.outlier).sum() +1)
        )


    plt.plot(recall, precisions)
    plt.show()

#%% Test DOCTOR strategy

accuracy_list = []
outlier_detection_df_list = []
for support_images, support_labels, query_images, query_labels, _ in tqdm(train_loader):
    model.process_support_set(support_images.cuda(), support_labels.cuda())
    predictions = model(query_images.cuda())

    accuracy_list.append((
            torch.max(
                predictions[:n_way*n_query].detach().data,
                1,
            )[1]
            == query_labels[:n_way*n_query].cuda()
        ).sum().item()/(n_way*n_query))

    outlier_detection_df_list.append(pd.DataFrame({
        "outlier": (n_way*n_query) * [False] + (n_way*n_query) * [True],
        "outlier_score": get_pseudo_renyi_entropy(predictions)
    }))

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

plot_roc(outlier_detection_df)


#%% Test LocalOutlierFactor

outlier_detection_df_list = []
for support_images, support_labels, query_images, query_labels, _ in tqdm(train_loader):
    support_features = model.backbone(support_images.cuda())
    query_features = model.backbone(query_images.cuda())

    clustering = LocalOutlierFactor(n_neighbors=3, novelty=True, metric="euclidean").fit(support_features.detach().cpu())

    outlier_detection_df_list.append(pd.DataFrame({
        "outlier": (n_way*n_query) * [False] + (n_way*n_query) * [True],
        "outlier_score": clustering.decision_function(query_features.detach().cpu())
    }))

outlier_detection_df = pd.concat(outlier_detection_df_list, ignore_index=True)

plot_roc(outlier_detection_df)

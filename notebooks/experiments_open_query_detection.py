#%%
from pathlib import Path
from statistics import mean

import pandas as pd
import torch
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.cifar import FewShotCIFAR100
from src.constants import (
    MINI_IMAGENET_ROOT_DIR,
    MINI_IMAGENET_SPECS_DIR,
    CIFAR_SPECS_DIR,
    CIFAR_ROOT_DIR,
    BACKBONES,
)
from src.inference_protonet import InferenceProtoNet
from src.mini_imagenet import MiniImageNet
from src.open_query_sampler import OpenQuerySampler

from src.utils import (
    set_random_seed,
    create_dataloader,
    plot_episode,
    plot_roc,
    get_pseudo_renyi_entropy,
)

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

set_random_seed(random_seed)

DATASET_CHOICE = "cifar"
# DATASET_CHOICE = "mini_imagenet"
BACKBONE_CHOICE = "resnet12"

model_weights = Path("data/models") / f"{BACKBONE_CHOICE}_{DATASET_CHOICE}_episodic.tar"

#%%
def get_cifar_set(split):
    return FewShotCIFAR100(
        root=CIFAR_ROOT_DIR,
        specs_file=CIFAR_SPECS_DIR / f"{split}.json",
        training=False,
    )


def get_mini_imagenet_set(split):
    return MiniImageNet(
        root=MINI_IMAGENET_ROOT_DIR,
        specs_file=MINI_IMAGENET_SPECS_DIR / f"{split}_images.csv",
        training=False,
    )


def get_test_loader(dataset_name):
    if dataset_name == "cifar":
        dataset = get_cifar_set("test")
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set("test")
    else:
        raise NotImplementedError("I don't know this dataset.")

    sampler = OpenQuerySampler(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks_per_epoch,
    )
    return create_dataloader(dataset, sampler, n_workers)


def get_train_loader(dataset_name, batch_size=1024):
    if dataset_name == "cifar":
        train_set = get_cifar_set("train")
    elif dataset_name == "mini_imagenet":
        train_set = get_mini_imagenet_set("train")
    else:
        raise NotImplementedError("I don't know this dataset.")

    return DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )


def get_inference_model(backbone, weights_path, train_loader):
    # We learnt that this custom ProtoNet gives better ROC curve (can be checked again later)
    inference_model = InferenceProtoNet(backbone, train_loader=train_loader).cuda()
    inference_model.load_state_dict(torch.load(weights_path))
    inference_model.eval()

    return inference_model


#%%
data_loader = get_test_loader(DATASET_CHOICE)

model = get_inference_model(
    BACKBONES[BACKBONE_CHOICE](),
    model_weights,
    get_train_loader(
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

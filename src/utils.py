import random
from pathlib import Path
from typing import Optional

import torchvision
from easyfsl.data_tools import TaskSampler
from easyfsl.methods import AbstractMetaLearner
import seaborn as sns
import numpy as np
import torch
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.datasets import FewShotCIFAR100, MiniImageNet

from src.constants import (
    BACKBONES,
    FEW_SHOT_METHODS,
    CIFAR_ROOT_DIR,
    CIFAR_SPECS_DIR,
    MINI_IMAGENET_ROOT_DIR,
    MINI_IMAGENET_SPECS_DIR,
)
from src.inference_protonet import InferenceProtoNet
from src.open_query_sampler import OpenQuerySampler


def set_random_seed(seed: int):
    """
    Set random, numpy and torch random seed, for reproducibility of the training
    Args:
        seed: defined random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed : {seed}")


def create_dataloader(dataset: Dataset, sampler: TaskSampler, n_workers: int):
    """
    Create a torch dataloader of tasks from the input dataset sampled according
    to the input tensor.
    Args:
        dataset: dataset from which to sample tasks
        sampler: task sampler, must implement an episodic_collate_fn method
        n_workers: number of workers of the dataloader

    Returns:
        a dataloader of tasks
    """
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=sampler.episodic_collate_fn,
    )


def build_model(
    backbone: str,
    feature_dimension: int,
    method: str,
    device: str,
    pretrained_weights: Optional[Path] = None,
) -> AbstractMetaLearner:
    """
    Build a meta-learner and cast it on the appropriate device
    Args:
        backbone: backbone of the model to build. Must be a key of constants.BACKBONES.
        feature_dimension: dimension of the feature space
        method: few-shot learning method to use
        device: device on which to put the model
        pretrained_weights: if you want to use pretrained_weights for the backbone

    Returns:
        a PrototypicalNetworks
    """
    convolutional_network = BACKBONES[backbone](num_classes=feature_dimension)

    model = FEW_SHOT_METHODS[method](convolutional_network).to(device)

    if pretrained_weights is not None:
        model.load_state_dict(torch.load(pretrained_weights))

    return model


def plot_episode(support_images, query_images):
    """
    Plot images of an episode, separating support and query images.
    Args:
        support_images (torch.Tensor): tensor of multiple-channel support images
        query_images (torch.Tensor): tensor of multiple-channel query images
    """

    def matplotlib_imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    support_grid = torchvision.utils.make_grid(support_images)
    matplotlib_imshow(support_grid)
    plt.title("support images")
    plt.show()
    query_grid = torchvision.utils.make_grid(query_images)
    plt.title("query images")
    matplotlib_imshow(query_grid)
    plt.show()


def plot_roc(outliers_df, title):
    fp_rate, tp_rate, _ = roc_curve(outliers_df.outlier, -outliers_df.outlier_score)

    plt.plot(fp_rate, tp_rate)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.show()

    return auc(fp_rate, tp_rate)


def plot_twin_hist(outliers_df, title):
    sns.histplot(data=outliers_df, x="outlier_score", hue="outlier")
    plt.title(title)
    plt.show()


def get_pseudo_renyi_entropy(predictions: torch.Tensor) -> torch.Tensor:
    return (
        torch.pow(nn.functional.softmax(predictions, dim=1), 2)
        .sum(dim=1)
        .detach()
        .cpu()
    )


def get_shannon_entropy(predictions: torch.Tensor) -> torch.Tensor:
    soft_prediction = nn.functional.softmax(predictions, dim=1)
    return (soft_prediction * torch.log(soft_prediction)).sum(dim=1).detach().cpu()


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


def get_task_loader(
    dataset_name, n_way, n_shot, n_query, n_tasks, split="test", n_workers=12
):
    if dataset_name == "cifar":
        dataset = get_cifar_set(split)
    elif dataset_name == "mini_imagenet":
        dataset = get_mini_imagenet_set(split)
    else:
        raise NotImplementedError("I don't know this dataset.")

    sampler = OpenQuerySampler(
        dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_tasks=n_tasks,
    )
    return create_dataloader(dataset, sampler, n_workers)


def get_classic_loader(dataset_name, split="train", batch_size=1024, n_workers=12):
    if dataset_name == "cifar":
        train_set = get_cifar_set(split)
    elif dataset_name == "mini_imagenet":
        train_set = get_mini_imagenet_set(split)
    else:
        raise NotImplementedError("I don't know this dataset.")

    return DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
    )


def get_inference_model(
    backbone, weights_path, align_train=True, train_loader=None, device="cuda"
):
    # We learnt that this custom ProtoNet gives better ROC curve (can be checked again later)
    inference_model = InferenceProtoNet(
        backbone, align_train=align_train, train_loader=train_loader
    ).to(device)
    inference_model.load_state_dict(torch.load(weights_path))
    inference_model.eval()

    return inference_model


def compute_features(feature_extractor: nn.Module, loader: DataLoader, device="cuda"):
    with torch.no_grad():
        all_features = []
        all_labels = []
        for images, labels in tqdm(loader, unit="batch"):
            all_features.append(feature_extractor(images.to(device)).data)
            all_labels.append(labels)

    return (
        torch.cat(all_features, dim=0).cpu().numpy(),
        torch.cat(all_labels, dim=0).cpu().numpy(),
    )


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

    plot_twin_hist(outliers_df, title=title)

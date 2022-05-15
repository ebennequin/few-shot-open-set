import random
import inspect
from typing import Tuple, Dict, Optional, Any, List
import numpy as np
import sklearn
import torch
from loguru import logger
from numpy import ndarray
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from src.models import __dict__ as BACKBONES
import os
from argparse import Namespace
from src.utils.data_fetchers import get_classic_loader
from collections import OrderedDict, defaultdict
import argparse
import torch.distributed as dist
from loguru import logger
import itertools


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


def tensor_product(left_tensor: torch.Tensor, right_tensor: torch.Tensor):
    """
    Args:
        left_tensor: shape (n, l)
        right_tensor: shape (m, l)

    Returns:
        their tensor product of shape (n, m, l).
        result[i,j,k] = left_tensor[i, k] * right_tensor[j, k]
    """
    return torch.matmul(
        left_tensor.T.unsqueeze(2), right_tensor.T.unsqueeze(1)
    ).permute((1, 2, 0))


def merge_from_dict(args, dict_: Dict):
    for key, value in dict_.items():
        setattr(args, key, value)
        # if isinstance(value, dict) and not any([isinstance(key, int) for key in value.keys()]):
        # setattr(args, key, Namespace())
        # merge_from_dict(getattr(args, key), value)
        # else:
        #     setattr(args, key, value)


def compute_features(
    feature_extractor: nn.Module,
    loader: DataLoader,
    split: str,
    device="cuda",
    keep_all_train_features=False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    with torch.no_grad():
        if split == "val" or split == "test" or keep_all_train_features:
            all_features = defaultdict(list)
            all_labels = []
            for images, labels in tqdm(loader, unit="batch"):
                feat = feature_extractor(images.to(device))
                all_features.append(feat.cpu())
                all_labels.append(labels)

            all_features = torch.cat(all_features, dim=0)
            return (
                all_features,
                torch.cat(all_labels, dim=0),
            )
        else:
            mean = 0.
            var = 0.
            N = 1.0
            for images, labels in tqdm(loader, unit="batch"):
                feats = feature_extractor(images.to(device))
                for new_sample in feats.cpu():
                    if N == 1:
                        mean = new_sample
                    else:
                        var = incremental_var(
                            var, mean, new_sample, N
                        )  # [d,]
                        mean = incremental_mean(
                            mean, new_sample, N
                        )  # [d,]
                    N += 1
            train_feats = torch.stack([mean, var], 0)  # [2, d]
            return train_feats, None


def incremental_mean(old_mean: Tensor, new_sample: Tensor, n: int):
    new_mean = 1 / n * (new_sample + (n - 1) * old_mean)
    return new_mean


def incremental_var(old_var: Tensor, old_mean: Tensor, new_sample: Tensor, n: int):
    new_var = (n - 2) / (n - 1) * (old_var) + 1 / n * (new_sample - old_mean) ** 2
    return new_var


def strip_prefix(state_dict: OrderedDict, prefix: str):
    return OrderedDict(
        [
            (k[len(prefix) :] if k.startswith(prefix) else k, v)
            for k, v in state_dict.items()
        ]
    )


def load_model(
    args,
    backbone: str,
    weights: Optional[Path],
    dataset_name: str,
    device: torch.device,
    num_classes: int = None,
):
    logger.info("Fetching data...")
    if num_classes is None:
        train_dataset, _, _ = get_classic_loader(
            args, dataset_name, split="train", batch_size=10
        )
        num_classes = len(np.unique(train_dataset.labels))

    logger.info("Building model...")
    feature_extractor = BACKBONES[backbone](
        num_classes=num_classes, pretrained=True
    ).to(device)

    if weights is not None:
        state_dict = torch.load(weights, map_location=device)
        if "state_dict" in state_dict:
            state_dict = strip_prefix(state_dict["state_dict"], "module.")
        elif "params" in state_dict:
            state_dict = strip_prefix(state_dict["params"], "encoder.")
            state_dict = strip_prefix(state_dict, "module.")
        else:
            state_dict = strip_prefix(state_dict, "backbone.")

        missing_keys, unexpected = feature_extractor.load_state_dict(
            state_dict, strict=False
        )
        logger.info(f"Loaded weights from {weights}")
        logger.info(f"Missing keys {missing_keys}")
        logger.info(f"Unexpected keys {unexpected}")

    feature_extractor.eval()

    return feature_extractor


def main_process(args: argparse.Namespace) -> bool:
    if args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            return True
        else:
            return False
    else:
        return True


def setup(args: argparse.Namespace, rank: int, world_size: int) -> None:
    """
    Used for distributed learning
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(args.port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    """
    Used for distributed learning
    """
    dist.destroy_process_group()


def find_free_port() -> int:
    """
    Used for distributed learning
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def get_modules_to_try(
    args, module_group: str, module_name: str, module_pool: Dict[str, Any], tune: bool
):

    modules_to_try: List[Any] = []
    # logger.warning(module_pool)
    module_dict = eval(f"args.{module_group}")

    if tune:
        logger.warning(f"Tuning over {module_group} activated")
        if module_name in eval(f"args.{module_group}"):
            shot = (
                args.n_shot if args.n_shot in module_dict[module_name]["default"] else 1
            )
            module_args = module_dict[module_name]["default"][shot]  # take default args
            if "tuning" in module_dict[module_name]:
                params2tune = module_dict[module_name]["tuning"]["hparams2tune"]
                shot = (
                    args.n_shot
                    if args.n_shot
                    in module_dict[module_name]["tuning"]["hparam_values"]
                    else 1
                )
                values2tune = module_dict[module_name]["tuning"]["hparam_values"][shot]
                values_combinations = itertools.product(*values2tune)
                for some_combin in values_combinations:
                    # Override default args
                    for k, v in zip(params2tune, some_combin):
                        module_args[k] = v
                    if (
                        "args"
                        in inspect.getfullargspec(
                            module_pool[module_name].__init__
                        ).args
                    ):
                        module_args["args"] = args
                    modules_to_try.append(module_pool[module_name](**module_args))
            else:
                logger.warning(
                    f"Module {module_name} has no specified grid to search over. Using default arguments."
                )
                modules_to_try.append(module_pool[module_name](**module_args))
        else:
            modules_to_try.append(module_pool[module_name]())
    else:
        module_args = {}
        if module_name in module_dict:
            shot = (
                args.n_shot if args.n_shot in module_dict[module_name]["default"] else 1
            )
            module_args = module_dict[module_name]["default"][shot]  # take default args
        if "args" in inspect.getfullargspec(module_pool[module_name].__init__).args:
            module_args["args"] = args
        modules_to_try = [module_pool[module_name](**module_args)]
    return modules_to_try


def normalize(features: Dict[int, ndarray]) -> Dict[int, ndarray]:
    return {k: sklearn.preprocessing.normalize(v, axis=1) for k, v in features.items()}

from typing import Dict, Tuple
from loguru import logger
import pandas as pd
import torch
from numpy import ndarray
from torch import nn
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(
        self,
        features_dict: Dict[int, ndarray],
    ):
        """
        features_dict[class][layer] = [tensor1, tensor2, ..., tensorN]
        """
        self.labels = []
        self.data = []
        for class_ in features_dict:
            all_features = list(features_dict[class_].values())
            n_samples = len(all_features[0])
            assert all([len(x) == n_samples for x in all_features])
            if n_samples >= 11:  # filter out classes with too few samples
                self.labels += [class_] * n_samples
                for sample_tuple in zip(*all_features):
                    sample_dic = {i: x for i, x in enumerate(sample_tuple)}
                    self.data.append(sample_dic)
            else:
                logger.warning(
                    f"Filtered out class {class_} because only contains {n_samples} samples"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.data[item.long()], self.labels[item.long()]

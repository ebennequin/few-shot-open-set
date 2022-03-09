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
        Build a dataset yielding feature vectors from a dictionary describing those vectors and
        their labels.
        Take numpy arrays as input, but yields torch tensors. Not very clean but fits our current
        needs.
        Args:
            features_dict: each key is an integer label, each value is a
                (n_samples, feature_dimension) numpy array containing the feature vectors for
                images with this label
            features_to_center_on: a 1-dim feature vector of length feature_dimension
        """
        self.labels = []
        self.data = []
        for class_ in features_dict:
            all_features = list(features_dict[class_].values())
            n_samples = len(all_features[0])
            assert all([len(x) == n_samples for x in all_features])
            self.labels += [class_] * n_samples
            for sample_tuple in zip(*all_features):
                sample_dic = {i: x for i, x in enumerate(sample_tuple)}
                self.data.append(sample_dic)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.data[item.long()], self.labels[item.long()]

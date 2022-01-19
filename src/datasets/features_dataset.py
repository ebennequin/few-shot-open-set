from typing import Dict, Tuple

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
        self.data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "label": k,
                        "features": list(
                            v,
                        ),
                    }
                )
                for k, v in features_dict.items()
            ],
            ignore_index=True,
        )
        self.labels = list(self.data.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, int]:
        item_data = self.data.loc[int(item)]
        return item_data.features, item_data.label

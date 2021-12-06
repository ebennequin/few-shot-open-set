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
        features_to_center_on: torch.Tensor = None,
    ):
        self.features_centered_on = (
            features_to_center_on if features_to_center_on is not None else 0.0
        )
        self.data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "label": k,
                        "features": list(
                            nn.functional.normalize(
                                torch.from_numpy(v) - self.features_centered_on, dim=1
                            )
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

    def __getitem__(self, item) -> Tuple[int, torch.Tensor]:
        item_data = self.data.loc[int(item)]
        return item_data.features, item_data.label

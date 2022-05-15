from typing import Dict, Tuple
from loguru import logger
import torch
from numpy import ndarray
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(
        self,
        features_dict: Dict[int, ndarray],
    ):
        """
        features_dict[class] = [tensor1, tensor2, ..., tensorN]
        """
        self.labels = []
        self.data = []
        for class_ in features_dict:
            all_features = features_dict[class_]
            n_samples = len(all_features)
            if n_samples >= 15:  # filter out classes with too few samples
                self.labels += [class_] * n_samples
                for tensor in zip(all_features):
                    self.data.append(tensor[0])
            else:
                logger.warning(
                    f"Filtered out class {class_} because only contains {n_samples} samples"
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.data[item.long()], self.labels[item.long()]

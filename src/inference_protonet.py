from typing import Optional

import torch
from easyfsl.methods import PrototypicalNetworks
from easyfsl.utils import compute_prototypes
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class InferenceProtoNet(PrototypicalNetworks):
    def __init__(
        self, *args, train_loader: DataLoader = None, align_train: bool = True
    ):
        """
        Following Bertinetto's inference strategy.
        Compute the whole train set's mean feature vector.
        At inference time, we center all support and query features around this mean,
        and normalize them.
        """
        super().__init__(*args)
        self.cuda()
        self.train_set_feature_mean = 0.0
        if align_train:
            self.get_dataset_feature_mean(train_loader)

    def process_support_set(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        self.prototypes = compute_prototypes(
            nn.functional.normalize(
                self.backbone(support_images).data - self.train_set_feature_mean, dim=1
            ),
            support_labels,
        )

    def forward(
        self,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        query_features = nn.functional.normalize(
            self.backbone(query_images.cuda()).data - self.train_set_feature_mean, dim=1
        )
        return -torch.cdist(query_features, self.prototypes)

    def get_dataset_feature_mean(self, data_loader: DataLoader):
        if data_loader is None:
            raise ValueError(
                "You forgot to provide a data loader for the train set. "
                "If you don't want to align on the train set average feature vector, "
                "specify align_train=False."
            )
        logger.info("Extracting features from all training set images...")
        self.eval()
        with torch.no_grad():
            all_features = []
            for images, _ in tqdm(data_loader, unit="batch"):
                all_features.append(self.backbone(images.cuda()).data)

        logger.info("Storing the average feature vector...")
        self.train_set_feature_mean = torch.cat(all_features, dim=0).mean(0)

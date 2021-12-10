"""
Utils for quick fetching of nn.Module objects.
"""

from pathlib import Path
from typing import Optional

import torch
from easyfsl.methods import AbstractMetaLearner

from src.constants import BACKBONES, FEW_SHOT_METHODS
from src.inference_protonet import InferenceProtoNet


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

import torch
from torch import nn


def sample_label_from_potential(potential: torch.Tensor) -> int:
    """
    Randomly sample a new label with the probability distribution obtained from normalized
    label potentials
    Args:
        potential: the current potentials for to-be-sampled labels, given already-sampled
            labels of this episode
    Returns:
        int: next sampled label
    """

    return int(torch.multinomial(nn.functional.normalize(potential, p=1, dim=0), 1))

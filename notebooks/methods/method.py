import argparse
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class FSmethod(nn.Module):
    '''
    Abstract class for few-shot methods
    '''
    def __init__(self, args: argparse.Namespace):
        super(FSmethod, self).__init__()

    def forward(self,
                feat_s: Tensor,
                feat_q: Tensor,
                y_s: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        returns:
            soft_preds: Tensor of shape [n_query, K], where K is the number of classes in the task,
                        representing the soft predictions of the method for the input query samples. 
        '''
        raise NotImplementedError
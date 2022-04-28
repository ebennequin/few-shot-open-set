import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Tuple, List, Dict
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_fn


class AllInOne:
    """
    Abstract class for few-shot methods
    """

    def __init__(self):
        super().__init__()
        self.works_on_features = (
            False  # by default most all-in-one methods need raw features and model
        )

    def compute_auc(self, outlierness, **kwargs):
        fp_rate, tp_rate, thresholds = roc_curve(
            kwargs["outliers"].numpy(), outlierness.cpu().numpy()
        )
        return auc_fn(fp_rate, tp_rate)

    def clear(self):
        pass

    def __call__(
        self, support_features: Tensor, query_features: Tensor, support_labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            support_features: support features
            query_features: query features
            support_labels: support labels

        Returns:
            support_soft_predictions: Tensor of shape [n_query, K], where K is the number of classes
                in the task, representing the soft predictions of the method for support samples.
            query_soft_predictions: Tensor of shape [n_query, K], where K is the number of classes
                in the task, representing the soft predictions of the method for query samples.
            outlier_scores: Tensor of shape [n_query,], where K is the number of classes
                in the task, representing the soft predictions of the method for query samples.
        """
        raise NotImplementedError

    @classmethod
    def from_cli_args(cls, args):
        signature = inspect.signature(cls.__init__)
        return cls(
            **{k: v for k, v in args._get_kwargs() if k in signature.parameters.keys()},
        )

    def __str__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if "args" in arg_names:
            arg_names.remove("args")
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

    def __repr__(self):
        arg_names = list(inspect.signature(self.__init__).parameters)
        if "args" in arg_names:
            arg_names.remove("args")
        if len(arg_names):
            args = [f"{k}={getattr(self, k)}" for k in arg_names]
            return f"{type(self).__name__}({','.join(args)})"
        else:
            return type(self).__name__

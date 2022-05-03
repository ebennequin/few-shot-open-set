from pathlib import Path
import numpy as np
from itertools import cycle
from collections import defaultdict
from functools import partial
from loguru import logger
from typing import Any, List, Tuple
from .csv_plotter import pretty, Plotter, my_default_dict
import torch
import matplotlib.pyplot as plt
import argparse
import json
import os
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as auc_fn


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--exp", type=str, help="Name of the experiment")
    parser.add_argument("--use_pretty", type=str2bool, default=True)
    parser.add_argument("--latex", type=str2bool, default=True)
    parser.add_argument(
        "--filters",
        type=str,
        nargs="+",
        default=[],
        help="Format: n_query=5 n_shot=1 ...",
    )
    parser.add_argument(
        "--groupby",
        type=str,
        nargs="+",
        help="Defines the methods compared. Ex: postpool_transforms",
    )

    args = parser.parse_args()
    return args


class TorchPlotter(Plotter):
    """
    An abstract plotter.
    """

    def fit(self, **kwargs):
        """
        Reads metrics in the folder and fill the dictionnary of metrics.
        At the end of this function, metric_dic should be filled as:

            metric_dic[method][metric_name] = torch.Tensor
        """
        # if kwargs['use_pretty']:
        #     process_dic = pretty
        # else:
        #     process_dic = my_default_dict(lambda x: x)

        #  ===== Recover all csv result files =====
        p = Path("results") / kwargs["exp"]
        torch_files = list(
            p.glob("**/*.pt")
        )
        torch_files.sort()
        assert len(torch_files)

        for file in torch_files:

            # ==== Make sure this is a relevant experiment ====

            filters = [x.split("=") for x in kwargs["filters"]]
            keep = True
            with open(Path(file.parent) / 'config.json', 'r') as f:
                config = json.load(f)
            for key, value in filters:
                expected_type = type(config[key])
                cast_value = np.array([value]).astype(expected_type)[0]
                if config[key] != cast_value:
                    keep = False

            if keep:
                array_name = file.stem
                # method = config[args.group_by]
                method = str(file.parts[-2])
                tensor = torch.load(file)
                self.metric_dic[method][array_name] = tensor

        self.out_dir = Path("plots") / kwargs["exp"] / "-".join(kwargs["filters"])

    def plot(self, **kwargs):

        for method in self.metric_dic:
            inliers = ~ self.metric_dic[method]['outliers'].bool()
            n_tasks, n_q, K = self.metric_dic[method]['probas_q'].size()

            flat_probs = self.metric_dic[method]['probas_q'].view(-1, K)
            # flat_maxprobs = flat_probs.max(-1).values
            flat_maxprobs = - (flat_probs * torch.log(flat_probs)).sum(-1)
            all_inliers = inliers.view(-1)

            # fig = plt.Figure()

            plt.hist(flat_maxprobs[all_inliers].numpy(), color='green', density=True, bins=100, alpha=0.4)
            plt.hist(flat_maxprobs[~all_inliers].numpy(), color='red', density=True, bins=100, alpha=0.4)
            fp_rate, tp_rate, _ = roc_curve((~all_inliers).numpy(), flat_maxprobs.numpy())
            plt.title(f"ROCAUC={np.round(auc_fn(fp_rate, tp_rate), 2)}")
            plt.ylim(0, 40)


            self.out_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(self.out_dir / f"{method}.pdf", dpi=300, bbox_inches="tight")
            plt.clf()


if __name__ == "__main__":
    args = parse_args()
    if args.latex:
        logger.info("Activating latex")
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
            }
        )
        # for Palatino and other serif fonts use:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Palatino"],
            }
        )

    plotter = TorchPlotter()
    plotter.fit(**vars(args))
    plotter.plot()

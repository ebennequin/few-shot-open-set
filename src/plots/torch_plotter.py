from pathlib import Path
import numpy as np
from itertools import cycle
from collections import defaultdict
from functools import partial
from loguru import logger
from typing import Any, List, Tuple
from .csv_plotter import pretty, Plotter, my_default_dict
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
import os
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc as auc_fn

colors = ["#0f6300", "#bd0d9e"]


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

        #  ===== Recover all csv result files =====
        p = Path("results") / kwargs["exp"]
        torch_files = list(p.glob("**/*.pt"))
        torch_files.sort()
        assert len(torch_files)

        self.baseline = {}

        for file in torch_files:
            # ==== Make sure this is a relevant experiment ====

            filters = [x.split("=") for x in kwargs["filters"]]
            keep = True
            with open(Path(file.parent) / "config.json", "r") as f:
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
                if "SimpleShot" in method:
                    self.baseline[array_name] = tensor
                else:
                    self.metric_dic[method][array_name] = tensor

        self.out_dir = Path("plots") / kwargs["exp"] / "-".join(kwargs["filters"])

    def plot(self, **kwargs):
        inliers = ~self.baseline["outliers"].bool()
        n_tasks, n_q, K = self.baseline["probas_q"].size()
        flat_probs = self.baseline["probas_q"].view(-1, K)
        baseline_maxprobs = -(flat_probs * torch.log(flat_probs)).sum(-1)
        all_inliers = inliers.view(-1)
        fp_rate, tp_rate, _ = roc_curve(
            (~all_inliers).numpy(), baseline_maxprobs.numpy()
        )
        baseline_auroc = np.round(100 * auc_fn(fp_rate, tp_rate), 1)

        fig, axes = plt.subplots(
            figsize=(6, 3), ncols=len(self.metric_dic), sharey=True
        )
        methods = list(self.metric_dic.keys())
        methods.sort(reverse=True)
        for i, (method, ax) in enumerate(zip(methods, axes)):
            inliers = ~self.metric_dic[method]["outliers"].bool()
            n_tasks, n_q, K = self.metric_dic[method]["probas_q"].size()

            flat_probs = self.metric_dic[method]["probas_q"].view(-1, K)
            # flat_maxprobs = flat_probs.max(-1).values

            flat_maxprobs = -(flat_probs * torch.log(flat_probs)).sum(-1)

            sns.kdeplot(
                baseline_maxprobs[all_inliers].numpy(),
                color=colors[0],
                alpha=1.0,
                linestyle="--",
                ax=ax,
                label="Inliers at initialization",
            )
            sns.kdeplot(
                flat_maxprobs[all_inliers].numpy(),
                fill=True,
                color=colors[0],
                alpha=0.4,
                label="Inliers after inference",
                ax=ax,
            )

            sns.kdeplot(
                baseline_maxprobs[~all_inliers].numpy(),
                color=colors[1],
                alpha=1.0,
                linestyle="--",
                ax=ax,
                label="Outliers at initialization",
            )
            sns.kdeplot(
                flat_maxprobs[~all_inliers].numpy(),
                fill=True,
                color=colors[1],
                alpha=0.4,
                label="Outliers after inference",
                ax=ax,
            )

            fp_rate, tp_rate, _ = roc_curve(
                (~all_inliers).numpy(), flat_maxprobs.numpy()
            )
            # ax.set_xlim(0, 1.7)
            # ax.set_ylim(0, 2.1)
            ax.set_xlabel(r"Closed-set entropy (nats)")
            ax.set_ylabel(r"Normalized frequency")
            # else:
            res = np.round(100 * auc_fn(fp_rate, tp_rate), 1)
            delta = np.round(res - baseline_auroc, 1)
            sign = r"\uparrow" if delta >= 0 else r"\downarrow"
            ax.set_title(
                rf'\textbf{{{pretty[method.split("(")[0]]}}}'
                "\n"
                rf"AUROC=${res}$ (${sign} {abs(delta)}$)",
                y=0.97,
            )
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            if i == 1:
                ax.spines["left"].set_visible(False)
                ax.set_yticks([])
        plt.subplots_adjust(wspace=0.05)
        plt.legend(
            frameon=False,
            loc="center",
            bbox_to_anchor=[0.0, 1.3],  # bottom-right
            ncol=2,
        )

        self.out_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            self.out_dir / f"entropy_histograms.pdf", dpi=300, bbox_inches="tight"
        )
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

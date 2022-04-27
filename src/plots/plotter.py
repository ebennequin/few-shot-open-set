from pathlib import Path
from itertools import cycle
from collections import defaultdict
from functools import partial
from typing import Any, List, Tuple
from loguru import logger

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

colors = [
    "g",
    "b",
    "m",
    "y",
    "k",
    "chartreuse",
    "coral",
    "gold",
    "lavender",
    "silver",
    "tan",
    "teal",
    "wheat",
    "orchid",
    "orange",
    "tomato",
]

styles = ["--", "-.", ":", "-"]


class Plotter:
    """
    An abstract plotter.
    """

    def __init__(
        self,
        figsize=[10, 10],
        fontsize=12,
        fontfamily="sans-serif",
        fontweight: str = "normal",
        dpi: int = 200,
        max_col: int = 1,
        out_extension: str = "png",
        out_dir=Path("./"),
    ):
        self.figsize = figsize
        self.fontsize = fontsize
        self.fontfamily = fontfamily
        self.fontweight = fontweight
        self.dpi = dpi
        self.max_col = max_col
        self.out_extension = out_extension
        self.out_dir = out_dir
        self.metric_dic = self.nested_default_dict(3, list)

    def fit(self, folder: Path):
        """
        Reads metrics in the folder and fill the dictionnary of metrics.
        At the end of this function, metric_dic should be filled as:

            metric_dic[metric_name][method] = {
                                                'x': ndarray [n_iter],
                                                'y': ndarray [n_iter],
                                                'pm': Optional[ndarray],
                                               }
        """
        raise NotImplementedError

    def plot(self, **kwargs):
        assert len(self.metric_dic)

        for metric in self.metric_dic:
            fig = plt.Figure(self.figsize, dpi=self.dpi)
            ax = fig.gca()
            for style, color, method in zip(
                cycle(styles), cycle(colors), self.metric_dic[metric]
            ):
                method_dic = self.metric_dic[metric][method]
                ax.plot(
                    method_dic["x"],
                    method_dic["y"],
                    label=method,
                    color=color,
                    linestyle=style,
                )
                if "pm" in method_dic:
                    ax.fill_between(
                        method_dic["x"],
                        method_dic["y"] - method_dic["pm"],
                        method_dic["y"] + method_dic["pm"],
                        color=color,
                        alpha=0.3,
                    )

            n_methods = len(self.metric_dic[metric])
            n_cols = min(self.max_col, n_methods)
            ax.legend(
                bbox_to_anchor=(0.5, 1.05),
                loc="center",
                ncol=n_cols,
                shadow=True,
                prop={"size": 6},
            )
            ax.set_xlabel(method_dic["xlabel"])
            ax.set_ylabel(metric)
            ax.grid(True)
            fig.tight_layout()
            save_path = Path("plots") / self.out_dir
            save_path.mkdir(parents=True, exist_ok=True)
            save_name = save_path / f"{metric}.{self.out_extension}"
            fig.savefig(save_name)
            logger.info(f"Figure saved at {save_name}")

    @staticmethod
    def nested_default_dict(depth: int, final_type: Any, i: int = 1):
        if i == depth:
            return defaultdict(final_type)
        fn = partial(
            Plotter.nested_default_dict, depth=depth, final_type=final_type, i=i + 1
        )
        return defaultdict(fn)

    @staticmethod
    def compute_confidence_interval(
        data: np.ndarray,
        axis=0,
        ignore_value=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 95% confidence interval
        :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
        :return: the 95% confidence interval for this data.
        """
        assert len(data)
        if ignore_value is None:
            valid = np.ones_like(data)
        else:
            valid = data != ignore_value
        m = np.sum(data * valid, axis=axis, keepdims=True) / valid.sum(
            axis=axis, keepdims=True
        )
        # np.mean(data, axis=axis)
        std = np.sqrt(((data - m) ** 2 * valid).sum(axis=axis) / valid.sum(axis=axis))
        # std = np.std(data, axis=axis)

        pm = 1.96 * (std / np.sqrt(valid.sum(axis=axis)))

        m = np.squeeze(m).astype(np.float64)
        pm = pm.astype(np.float64)

        return m, pm

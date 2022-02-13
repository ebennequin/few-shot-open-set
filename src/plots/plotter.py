from pathlib import Path
from itertools import cycle
from collections import defaultdict
from functools import partial
from typing import Any, List, Tuple
from loguru import logger

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

colors = ["g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
          'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

styles = ['--', '-.', ':', '-']


class Plotter:
    """
    An abstract plotter.
    """
    def __init__(self, figsize=[10, 10], fontsize=12, fontfamily='sans-serif',
                 fontweight='normal', dpi: int = 200, max_col: int = 1):
        self.figsize = figsize
        self.fontsize = fontsize
        self.fontfamily = fontfamily
        self.fontweight = fontweight
        self.dpi = dpi
        self.max_col = max_col
        # self.metric_dic = {}

    def fit(self, folder):
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

    def plot(self):

        for metric in self.metric_dic:
            fig = plt.Figure(self.figsize, dpi=self.dpi)
            ax = fig.gca()
            for style, color, method in zip(cycle(styles), cycle(colors), self.metric_dic[metric]):
                mean, std = compute_confidence_interval(y, axis=0)
                n_epochs = mean.shape[0]
                x = np.linspace(0, n_epochs - 1, (n_epochs)) * res_dic['log_freq']
                valid_iter = (mean != 0)

                label = method
                ax.plot(x[valid_iter], mean[valid_iter], label=label, color=color, linestyle=style)
                ax.fill_between(x[valid_iter], mean[valid_iter] - std[valid_iter],
                                mean[valid_iter] + std[valid_iter], color=color, alpha=0.3)

            n_cols = min(max_col, len(filenames_dic[metric]))
            ax.legend(bbox_to_anchor=(0.5, 1.05), loc='center', ncol=n_cols, shadow=True)
            ax.set_xlabel("Training iterations")
            ax.grid(True)
            fig.tight_layout()
            save_path = p / 'plots'
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / f'{metric}.png')
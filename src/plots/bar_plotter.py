from pathlib import Path
import numpy as np
from itertools import cycle
from collections import defaultdict
from functools import partial
from loguru import logger
from typing import Any, List, Tuple
from .plotter import Plotter
import pandas as pd
import argparse
from .csv_plotter import CSVPlotter, parse_args, pretty
import matplotlib.pyplot as plt
import os

colors = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber,
          CB91_Purple, CB91_Violet, 'r', 'm'] 


class BarPlotter(CSVPlotter):
    """
    An abstract plotter.
    """
    def plot(self, **kwargs):
        '''
        metric_dic[metric_name][method] = {
                                    'x': ndarray [n_points_found],
                                    'y': ndarray [n_points_found],
                                    'pm': Optional[ndarray],
                                   }
        '''
        fig = plt.Figure(figsize=(10, 13))
        ax = fig.gca()

        model_delta_x = {'RN-18': -0.3, 'RN-50': -0.15, 'EN-B4': 0., 'RN-101': 0.15, 'ViT-B': 0.3}

        metric_names = list(self.metric_dic.keys())
                
        for i, (metric_name, metric_dic) in enumerate(self.metric_dic):

            methods = list(metric_dic.keys())
            colors = {methods}
            labels = metric_dic[methods[0]]['x']
            offsets = np.linspace(-0.15, 0.15, len(labels))
            center = i
            bottoms = defaultdict(float)
            assert len(metric_dic) == 2, "Currently only support 2 methods at a time."
            for method, method_dic in metric_dic.items():
                for j, (label, value) in enumerate(zip(method_dic['x'], method_dic['y'])):
                    pos = center + offsets[i]
                    ax.bar([pos], [value], edgecolor='white', width=0.1, bottom=[bottoms[label]], label=label)
                    bottoms[label] += value

        # ax.text(pos - 0.065, bottom + 0.01, model, size=20)
        ax.set_xticks(len(metric_names))
        ax.set_xticklabels(metric_names)
        # ax.set_ylabel('Runtime / Batch (s)')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        fig.tight_layout()
        os.makedirs(self.out_dir, exist_ok=True)
        fig.savefig(self.out_dir / f'barplot.pdf', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    args = parse_args()
    if args.latex:
        logger.info("Activating latex")
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"]})
        # for Palatino and other serif fonts use:
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     "font.family": "serif",
        #     "font.serif": ["Palatino"],
        # })
    plotter = SpiderPlotter()
    plotter.fit(**vars(args))
    plotter.plot()
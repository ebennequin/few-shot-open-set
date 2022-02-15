from pathlib import Path
from itertools import cycle
from collections import defaultdict
from functools import partial
from typing import Any, List, Tuple
from loguru import logger
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
from .Plotter import Plotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument('--folder', type=str,
                        help='Folder to search')
    parser.add_argument('--reduce_by', type=str, default='random_seed')
    parser.add_argument('--simu_params', type=str, nargs='+',
                        default=['dataset', 'backbone'])

    args = parser.parse_args()
    return args


class NumpyPlotter(Plotter):
    """
    An abstract plotter.
    """

    def fit(self, folder: Path, **kwargs):
        """
        Reads metrics in the folder and fill the dictionnary of metrics.
        At the end of this function, metric_dic should be filled as:

            metric_dic[metric_name][method] = {
                                                'x': ndarray [n_iter],
                                                'y': ndarray [n_iter],
                                                'pm': Optional[ndarray],
                                               }
        """
        # Recover all files that match .npy pattern in folder/
        p = Path(folder)
        all_files = list(p.glob('**/*.npy'))

        if not len(all_files):
            print("No .pny files found in this subtree. Cancelling plotting operations.")
            return

        # Group files by metric name
        filenames_dic = self.nested_default_dict(4, str)
        for path in all_files:
            root = path.parent
            metric = path.stem
            with open(root / 'config.json') as f:
                config = json.load(f)
            fixed_key = tuple([config[key] for key in kwargs['simu_params']])
            reduce_key = config[kwargs['reduce_by']]
            filenames_dic[metric][fixed_key][reduce_key]['log_freq'] = config['log_freq']
            filenames_dic[metric][fixed_key][reduce_key]['path'] = path

        # Do one plot per metric
        for metric in filenames_dic:
            for simu_args in filenames_dic[metric]:
                values = []
                for _, res_dic in filenames_dic[metric][simu_args].items():
                    values.append(np.load(res_dic['path']))  # [N_iter]
                values = np.stack(values, axis=0)  # [#same_simu, N_iter]
                self.metric_dic[metric][simu_args]['y'], self.metric_dic[metric][simu_args]['pm'] = self.compute_confidence_interval(values)
                self.metric_dic[metric][simu_args]['x'] = np.arange(len(self.metric_dic[metric][simu_args]['y']))


if __name__ == "__main__":
    args = parse_args()
    plotter = NumpyPlotter()
    plotter.fit(**vars(args))
    plotter.plot()

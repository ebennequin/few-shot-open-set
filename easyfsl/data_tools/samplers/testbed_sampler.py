from pathlib import Path
from typing import Iterator, List

import pandas as pd

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.samplers import AbstractTaskSampler


class TestbedSampler(AbstractTaskSampler):
    def __init__(self, dataset: EasySet, testbed_csv: Path):
        self.testbed = pd.read_csv(testbed_csv, index_col=0).sort_values(
            ["task", "labels", "support"], ascending=[True, True, False]
        )
        # TODO: not cool at all, I need to find a way to identify an item as support or
        # query in episodic_collate_fn
        self.n_shot = (
            self.testbed.loc[self.testbed.task == 0]
            .groupby(["task", "labels"])
            .support.sum()
            .max()
        )
        self.n_way = self.testbed.loc[self.testbed.task == 0].labels.nunique()
        self.n_query = (
            self.testbed.loc[self.testbed.task == 0]
            .groupby(["task", "labels"])
            .support.count()
            .max()
            - self.n_shot
        )

    def __iter__(self) -> Iterator[List[int]]:
        return iter([list(task[1]) for task in self.testbed.groupby("task").image_id])

    def __len__(self):
        return self.testbed.task.nunique()

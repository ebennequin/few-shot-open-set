from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm

from easyfsl.data_tools import EasySet
from easyfsl.data_tools.dag_utils import build_wordnet_dag, reduce_to_leaves


class EasySemantics:
    def __init__(self, dataset: EasySet, path_to_wordnet: Path):
        """
        Get the Directed Acyclic Graph where the leafs are the classes of the dataset, and that
        defines the semantic hierarchy of classes following the Wordnet hierarchy.
        Args:
            dataset: an EasySet dataset
            path_to_wordnet: path to a txt file where each line contains an edge
                (parent class, child class) of the Wordnet hierarchy.
        """
        self.class_names = dataset.class_names
        self.n_items_per_class = {
            class_name: dataset.labels.count(class_id)
            for class_id, class_name in enumerate(dataset.class_names)
        }

        self.wordnet_dag = build_wordnet_dag(path_to_wordnet)
        self.dataset_dag = reduce_to_leaves(self.wordnet_dag, self.class_names)

    def get_semantic_distance(self, class_a: str, class_b: str) -> float:
        """
        Compute the Jiang and Conrath semantic distance between two classes of the dataset.
        It is defined as 2 log(C) - (log(A) + log(B)) where A (resp. B) is the number of images
        with label class_a (resp. class_b) in the dataset, and C is the number of images which label
        is a descendant of the lowest common ancestor of class_a and class_b in the semantic DAG.
        Note that the function is symmetric between class_a and class_b.
        Args:
            class_a: first class name (must be a node of the DAG)
            class_b: second class name (must be a node of the DAG)
        Returns:
            the Jiang and Conrath distance between class_a and class_b
        """
        if class_a == class_b:
            return 0.0

        lowest_common_ancestor = nx.lowest_common_ancestor(
            self.dataset_dag, class_a, class_b
        )
        spanned = [
            node
            for node in nx.algorithms.dag.descendants(
                self.dataset_dag, lowest_common_ancestor
            )
            if self.dataset_dag.out_degree[node] == 0
        ]

        population_common_ancestor = sum(
            [self.n_items_per_class[leave] for leave in spanned]
        )

        population_a = self.n_items_per_class[class_a]
        population_b = self.n_items_per_class[class_b]

        return 2 * np.log(population_common_ancestor) - (
            np.log(population_a) + np.log(population_b)
        )

    def get_semantic_distance_matrix(self) -> np.ndarray:
        """
        Compute the Jiang and Conrath semantic distance between all classes of the dataset.
        The distance between class_a and class_b is defined as 2 log(C) - (log(A) + log(B)) where
        A (resp. B) is the number of images with label class_a (resp. class_b) in the dataset,
        and C is the number of images which label is a descendant of the lowest common ancestor of
        class_a and class_b in the semantic DAG.

        Returns:
            symmetric square matrix of floats. Value at (i,j) is the semantic distance between
                classes i and j
        """
        distances = np.zeros((len(self.class_names), len(self.class_names)))

        for class_a in tqdm(range(len(self.class_names)), unit="classes"):
            for class_b in range(class_a, len(self.class_names)):
                distances[class_a, class_b] = self.get_semantic_distance(
                    self.class_names[class_a], self.class_names[class_b]
                )

        return distances + distances.T

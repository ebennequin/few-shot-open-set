from pathlib import Path
from typing import List

import networkx as nx


def build_wordnet_dag(path_to_wordnet: Path) -> nx.DiGraph:
    """
    Build the Wordnet hierarchy in the shape of a Direct Acyclic Graph (DAG).
    Args:
        path_to_wordnet: path to a txt file where each line contains an edge
            (parent class, child class) of the Wordnet hierarchy.

    Returns:
        a directed acyclic graph
    """
    whole_dag = nx.DiGraph()
    with open(path_to_wordnet) as file:
        for line in file:
            parent, child = line.rstrip().split(" ")
            whole_dag.add_edge(parent, child)

    return whole_dag


def reduce_to_leaves(dag: nx.DiGraph, leaves: List[str]) -> nx.DiGraph:
    """
    Form a sub-graph of the input Directed Acyclic Graph (DAG) reduced to the paths between the
    input DAG's root and every input leaves. "Tunnels" (i.e. nodes with exactly one parent and one
    child) are collapsed.
    Args:
        dag (nx.DiGraph): directed acyclic graph
        leaves (list[str]): expected leaves of the output DAG
    Returns:
        nx.DiGraph: subgraph formed by the union of shortest paths from each leave to the root
    """

    root = next(nx.lexicographical_topological_sort(dag))
    nodes = []

    for leave in leaves:
        nodes += nx.shortest_path(dag, source=root, target=leave)

    graph_reduced_to_leaves = nx.DiGraph(dag.subgraph(nodes))

    tunnels = [
        node
        for node in graph_reduced_to_leaves
        if graph_reduced_to_leaves.in_degree[node] == 1
        and graph_reduced_to_leaves.out_degree[node] == 1
    ]

    for tunnel_node in tunnels:
        graph_reduced_to_leaves.add_edge(
            next(graph_reduced_to_leaves.predecessors(tunnel_node)),
            next(graph_reduced_to_leaves.successors(tunnel_node)),
        )
        graph_reduced_to_leaves.remove_node(tunnel_node)

    return graph_reduced_to_leaves

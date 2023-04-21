import sys
from copy import deepcopy
from functools import reduce
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from week6_pagerank.utils import get_diff

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

def load_edges(which: str):
    return pd.read_csv(DATA_DIR / f"{which}.csv", names=["from", "_1", "to", "_2"])[["from", "to"]].values.tolist()


if __name__ == '__main__':
    dataset = "dolphins"
    graph = nx.Graph()
    edges = load_edges(dataset)
    nodes = set(reduce(lambda i, j: i + j, zip(*edges)))
    n_nodes = len(nodes)

    initial_rank = 1 / n_nodes
    graph.add_nodes_from(nodes, rank=initial_rank)
    graph.add_edges_from(edges)

    ranks = {}

    for key, node in graph.nodes(data=True):
        ranks[key] = node.get('rank')

    n_iter = 30
    beta = 0.8
    convergence = []
    for _ in range(n_iter):
        prev_ranks = deepcopy(ranks)
        for key, node in graph.nodes(data=True):
            rank_sum = 0.0
            neighbours = graph[key]
            for n in neighbours:
                if ranks[n] is not None:
                    outlinks = len(list(graph.neighbors(n)))
                    rank_sum += (1 / outlinks) * ranks[n]
            ranks[key] = beta * rank_sum + (1 - beta) * (1 / n_nodes)

        convergence.append(get_diff(prev_ranks, ranks))


    with open(f"{dataset}-basic-analysis.txt", 'w') as sys.stdout:
        pprint(ranks)

    plt.plot(convergence)
    plt.grid()
    plt.title("Convergence of Pagerank without Map-Reduce")
    plt.xlabel("Step")
    plt.title("Score Difference")
    plt.savefig(f"./images/{dataset}-conv.png")
    plt.close()

    plt.plot(convergence)
    plt.grid()
    plt.title("Convergence of Pagerank without Map-Reduce")
    plt.xlabel("Step")
    plt.title("Score Difference (log)")
    plt.yscale("log")
    plt.savefig(f"./images/{dataset}-conv-log.png")
    plt.close()




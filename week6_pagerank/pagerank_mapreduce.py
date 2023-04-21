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


def func_map(prev_ranks):
    ret = []
    for key, node in prev_ranks:
        neighbours = graph[key]
        score = (1 / len(neighbours)) * node['rank']
        ret += [
            (neighbour, score)
            for neighbour in neighbours
        ]
    return ret

def func_group(mapped_result):
    res = sorted(mapped_result)
    ret = {}
    for r in res:
        k, v = r
        if k not in ret:
            ret[k] = []
        ret[k].append(v)

    return ret


def func_reduce(grouped_result):
    return {
        k: sum(vs)
        for k, vs in grouped_result.items()
    }


def random_teleport(reduced_result, beta, n_nodes):
    return {
        k: beta * v + (1 - beta) * (1 / n_nodes)
        for k, v in reduced_result.items()
    }

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
        map_res = func_map(graph.nodes(data=True))
        group_res = func_group(map_res)
        reduce_res = func_reduce(group_res)
        random_teleported = random_teleport(reduce_res, beta, n_nodes)
        ranks = random_teleported
        for key, node in graph.nodes(data=True):
            node['rank'] = random_teleported[key]


        convergence.append(get_diff(prev_ranks, ranks))

    with open(f"{dataset}-mr-analysis.txt", 'w') as sys.stdout:
        out = sorted(graph.nodes.data(), key=lambda i: i[0])
        pprint(dict(map(lambda r: (r[0], r[1]['rank']), out)))

    plt.plot(convergence)
    plt.grid()
    plt.title("Convergence of Pagerank with Map-Reduce")
    plt.xlabel("Step")
    plt.title("Score Difference")
    plt.savefig(f"./images/{dataset}-mr-conv.png")
    plt.close()

    plt.plot(convergence)
    plt.grid()
    plt.title("Convergence of Pagerank with Map-Reduce")
    plt.xlabel("Step")
    plt.title("Score Difference (log)")
    plt.yscale("log")
    plt.savefig(f"./images/{dataset}-mr-conv-log.png")
    plt.close()

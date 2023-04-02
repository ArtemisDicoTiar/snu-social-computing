import sys
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nxcom

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"


def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


def main():
    df = pd.read_csv(DATA_DIR / "lesmis.csv", names=["from", "n0", "to", "n1"])

    nodes = set(df["from"])
    edges = df[["from", "to"]].values.tolist()

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    """
    plt.figure(figsize=(15, 15))
    nx.draw_networkx(
        graph,
        node_size=10,
        edge_color="#444444",
        alpha=0.5,
        with_labels=True,
        font_size=8
    )
    plt.show()
    """

    deg_centrality = nx.degree_centrality(graph)
    bet_centrality = nx.betweenness_centrality(graph)
    clo_centrality = nx.closeness_centrality(graph)
    eig_centrality = nx.eigenvector_centrality(graph)

    def print_sorted_items(itm: dict, which: str):
        itms = [(b, a) for a, b in itm.items()]
        itms.sort()
        itms.reverse()
        print(f"================ {which} ================")
        pprint(itms[:5])

    print_sorted_items(deg_centrality, "degree")
    print_sorted_items(bet_centrality, "betweenness")
    print_sorted_items(clo_centrality, "closeness")
    print_sorted_items(eig_centrality, "eigenvector")

    greedy_comms = sorted(nxcom.greedy_modularity_communities(graph), key=len, reverse=True)
    girnew_comms = sorted(nxcom.girvan_newman(graph), key=len, reverse=True)

    print("================ greedy-mod ================")
    greedy_mod = nxcom.modularity(graph, greedy_comms)
    print(f"len: {len(greedy_comms)}, mod: {greedy_mod}")
    pprint(greedy_comms)

    mods = []
    for girnew_comm in girnew_comms:
        print(f"================ girnew-mod grp: {len(girnew_comm)} ================")
        mod = nxcom.modularity(graph, girnew_comm)
        mods.append((len(girnew_comm), mod))
        print(f"len: {len(girnew_comm)}, mod: {mod}")
        pprint(girnew_comm)

    plt.plot(*zip(*mods), 'b-o')
    plt.axhline(y=greedy_mod, color='r', linestyle='-')

    max_x, max_y = sorted(mods, key=lambda i: i[1], reverse=True)[0]
    plt.text(70, greedy_mod, (5, round(greedy_mod, 3)), size=12)
    plt.text(max_x, max_y, (max_x, round(max_y, 3)), size=12)
    plt.grid()
    plt.legend(["girvan-newman", "greedy"])
    plt.title("Modularity - # of community")
    plt.xlabel("Number of community")
    plt.ylabel("Modularity")
    plt.savefig("./images/mod-com.png")
    plt.close()

    def plot_community_graph(g, comm, which: str):
        set_node_community(g, comm)
        set_edge_community(g)

        # Set community color for nodes
        node_color = [
            get_color(g.nodes[v]['community'])
            for v in g.nodes]

        # Set community color for internal edgese
        external = [
            (v, w) for v, w in g.edges
            if g.edges[v, w]['community'] == 0]
        internal = [
            (v, w) for v, w in g.edges
            if g.edges[v, w]['community'] > 0]
        internal_color = [
            get_color(g.edges[e]['community'])
            for e in internal]

        pos = nx.spring_layout(g)
        plt.rcParams.update({'figure.figsize': (15, 15)})
        # Draw external edges
        nx.draw_networkx(
            g, pos=pos, node_size=0,
            edgelist=external, edge_color="#333333", with_labels=True)
        # Draw nodes and internal edges
        nx.draw_networkx(
            g, pos=pos, node_color=node_color,
            edgelist=internal, edge_color=internal_color, with_labels=True)

        plt.title(which)
        plt.savefig(f"./images/{which}.png")
        plt.close()

    plot_community_graph(graph, comm=greedy_comms, which="greedy_modularity")
    for girnew_comm in girnew_comms:
        plot_community_graph(graph, comm=girnew_comm, which=f"girvan-newman grps:{len(girnew_comm)}")


if __name__ == '__main__':
    with open(f"lesmis-analysis.txt", 'w') as sys.stdout:
        main()

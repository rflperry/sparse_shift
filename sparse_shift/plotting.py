"""Tools for visualizing DAGs and more"""
# Authors: Ronan Perry
# License: MIT

import networkx as nx
import matplotlib.pyplot as plt


def plot_dag(
    adj,
    topological_sort=True,
    parent_adj=False,
    layout="circular",
    figsize=(5, 5),
    title=None,
    highlight_nodes=None,
    highlight_edges=None,
    labels=None,
    node_size=500,
):
    if parent_adj:
        adj = adj.T
 
    G = nx.convert_matrix.from_numpy_matrix(adj, create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)

    if layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = None
        # raise NotImplementedError(f'layout {layout} not a valid mode yet')

    edge_options = {
        "width": 3,
        "arrowstyle": "-|>",
        "arrowsize": 20,
        "alpha": 0.5,
        "arrows": True,
        "node_size": node_size,
    }

    labeldict = {}
    for i in range(adj.shape[0]):
        if labels is None:
            labeldict[i] = f"X{i+1}"
        else:
            labeldict[i] = labels[i]

    if highlight_edges is not None:
        if parent_adj:
            highlight_edges = highlight_edges.T
        black_edges = [edge for edge in G.edges() if ((adj[edge] == 1) and (highlight_edges[edge] == 0))]
        red_edges = [edge for edge in G.edges() if ((adj[edge] == 1) and (highlight_edges[edge] != 0))]
    else:
        red_edges = []
        black_edges = G.edges()

    if highlight_nodes is not None:
        node_colors = ['white' if highlight_nodes[node] == 0  else 'red' for node in G.nodes()]
    else:
        node_colors = 'white'

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        G, pos=pos,
        node_color=node_colors,
        ax=ax,
        node_size=node_size, alpha=0.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labeldict)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', ax=ax, **edge_options)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, edge_color='black', ax=ax, **edge_options)

    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000")
    if title is not None:
        plt.title(title)
    plt.box(False)

import networkx as nx
import numpy as np


def _graph2dag(graph):
    """Converts nx.Graph to an directed, acyclic form. Returns the adjancency matrix"""
    adj = nx.adj_matrix(graph).todense()
    adj = adj + adj.T
    adj = (adj != 0).astype(int)
    adj = np.tril(adj)

    assert nx.is_directed_acyclic_graph(nx.from_numpy_matrix(adj, create_using=nx.DiGraph))

    return adj


def erdos_renyi_dag(n, p, seed=None):
    """
    Simulates an Erdos Renyi random DAG on n vertices
    with expected degree p. Each node has the same expected
    degree.

    If p is an integer, it is the expected
    number of connected edges. Else, it is the expected degree
    fraction relative to n.
    """
    if isinstance(p, int):
        p = p / n
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)
    return _graph2dag(G)


def connected_erdos_renyi_dag(n, p, seed=None):
    """
    Simulates an Erdos Renyi random DAG on n vertices
    with expected degree p. Each node has the same expected
    degree and the graph is gauranteed connected, with
    a deterministic number of edges.

    If p is an integer, it is the expected
    number of connected edges. Else, it is the expected degree
    fraction relative to n.
    """
    if isinstance(p, float):
        p = p * n
        if int(p) != p:
            import warnings
            warnings.warn(f'Number of neighbors {p:.1f} will be rounded')

    G = nx.connected_watts_strogatz_graph(
        n, k=round(p), p=1 - 1/n, seed=seed
        )
    return _graph2dag(G)


def barabasi_albert_dag(n, p, seed=None):
    """
    Simulates an Barabasi Albert DAG on n vertices
    with expected degree p. The degree distribution follows
    a power law, and the graph is guaranteed to be connected.

    If p is an integer, it is the expected
    number of connected edges. Else, it is the expected degree
    fraction relative to n. Important, p must be <= 0.5
    or the integer equivalent to be guaranteed to succeed on all graphs.
    """
    if isinstance(p, int):
        p = p / n

    # BA model input m leads to K=(1+...+m) + m*(n-m) total edges
    # p = K
    m = 0.5*(2*n + 1 - np.sqrt(4*n**2 - 8*p*n**2 + 4*n + 1))
    if int(m) != m:
        import warnings
        warnings.warn(f'Number of neighbors {m:.1f} will be rounded')

    G = nx.barabasi_albert_graph(n, round(m), seed)
    return _graph2dag(G)


def complete_dag(n, p=None, seed=None):
    """
    Returns a complete DAG over n variables
    """
    G = np.ones((n, n)) - np.eye(n)
    return np.tril(G)

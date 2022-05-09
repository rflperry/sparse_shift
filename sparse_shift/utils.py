"""Tools for other functions and methods"""
import numpy as np
from causaldag import DAG, PDAG


def check_2d(X):
    if X is not None and X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def dags2mechanisms(dags):
    """
    Returns a dictionary of variable: mechanisms from
    a list of DAGs.
    """
    m = len(dags[0])
    mech_dict = {i: [] for i in range(m)}
    for dag in dags:
        for i, mech in enumerate(dag.T):  # Transpose to get parents
            mech_dict[i].append(mech)

    # remove duplicates
    for i in range(m):
        mech_dict[i] = np.unique(mech_dict[i], axis=0)

    return mech_dict


def create_causal_learn_dag(G):
    """Converts directed adj matrix G to causal graph"""
    from causallearn.graph.Dag import Dag
    from causallearn.graph.GraphNode import GraphNode

    n_vars = G.shape[0]
    node_names = [("X%d" % (i + 1)) for i in range(n_vars)]
    nodes = [GraphNode(name) for name in node_names]

    cl_dag = Dag(nodes)
    for i in range(n_vars):
        for j in range(n_vars):
            if G[i, j] != 0:
                cl_dag.add_directed_edge(nodes[i], nodes[j])

    return cl_dag


def create_causal_learn_cpdag(G):
    """Converts adj mat of cpdag to a causal learn graph object"""
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint
    from causallearn.graph.GeneralGraph import GeneralGraph
    from causallearn.graph.GraphNode import GraphNode

    n_vars = G.shape[0]
    node_names = [("X%d" % (i + 1)) for i in range(n_vars)]
    nodes = [GraphNode(name) for name in node_names]

    cl_cpdag = GeneralGraph(nodes)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if G[i, j] == 1 and G[j, i] == 1:
                cl_cpdag.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL))
            elif G[i, j] == 1 and G[j, i] == 0:
                cl_cpdag.add_edge(Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW))
            elif G[i, j] == 0 and G[j, i] == 1:
                cl_cpdag.add_edge(Edge(nodes[i], nodes[j], Endpoint.ARROW, Endpoint.TAIL))

    return cl_cpdag


def dag2cpdag(adj, targets=None):
    """Converts an adjacency matrix to the cpdag adjacency matrix, with potential interventions"""
    dag = DAG().from_amat(adj)
    cpdag = dag.cpdag()
    if targets is None:
        return cpdag.to_amat()[0]
    else:
        return dag.interventional_cpdag(
                [targets], cpdag=cpdag
            ).to_amat()[0]

def cpdag2dags(adj):
    """Converts a cpdag adjacency matrix to a list of all dags"""
    adj = np.asarray(adj)
    dags_elist = list(PDAG().from_amat(adj).all_dags())
    dags = []
    for elist in dags_elist:
        G = np.zeros(adj.shape)
        elist = np.asarray(list(elist))
        if len(elist) > 0:
            G[elist[:, 0], elist[:, 1]] = 1
        dags.append(G)
    
    return dags

"""
Useful causal-learn utils for reference

# Orients edges in a pdag, to find a dag (not necessarily possible)
causallearn.utils.PDAG2DAG import pdag2dag

# Returns the CPDAG of a DAG (the MEC!)
from causallearn.utils.DAG2CPDAG import dag2cpdag

# Checks if two dags are in the same MEC
from causallearn.utils.MECCheck import mec_check

# Runs meek's orientation rules over a DAG, with optional background
# knowledge. definite_meek examines definite unshielded triples
from causallearn.utils.PCUtils.Meek import meek, definite_meek

"""
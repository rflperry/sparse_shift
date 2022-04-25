import numpy as np


def dag_tpr(true_dag, cpdag):
    # np.testing.assert_array_equal(true_dag, np.tril(true_dag))
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    n_edges = np.sum(true_dag)
    return tp / n_edges


def dag_fpr(true_dag, cpdag):
    # np.testing.assert_array_equal(true_dag, np.tril(true_dag))
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0])
    n_edges = np.sum(true_dag)
    return fp / n_edges

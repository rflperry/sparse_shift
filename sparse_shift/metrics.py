import numpy as np


def dag_true_orientations(true_dag, cpdag):
    """Number of correctly oriented edges / number of edges"""
    # np.testing.assert_array_equal(true_dag, np.tril(true_dag))
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    n_edges = np.sum(true_dag)
    return tp / n_edges


def dag_false_orientations(true_dag, cpdag):
    """Number of falsely oriented edges / number of edges"""
    # np.testing.assert_array_equal(true_dag, np.tril(true_dag))
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0])
    n_edges = np.sum(true_dag)
    return fp / n_edges


def dag_precision(true_dag, cpdag):
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    fp = len(np.where((true_dag + cpdag.T - cpdag) == 2)[0])
    return tp / (tp + fp) if (tp + fp) > 0 else 1


def dag_recall(true_dag, cpdag):
    tp = len(np.where((true_dag + cpdag - cpdag.T) == 2)[0])
    return tp / np.sum(true_dag)


def average_precision_score(true_dag, pvalues_mat):
    """
    Computes average precision score from pvalue thresholds
    """
    from sparse_shift.utils import dag2cpdag, cpdag2dags
    thresholds = np.unique(pvalues_mat)
    dags = np.asarray(cpdag2dags(dag2cpdag(true_dag)))

    # ap_score = 0
    # prior_recall = 0

    precisions = []
    recalls = []

    for t in thresholds:
        axis = tuple(np.arange(1, pvalues_mat.ndim))
        n_changes = np.sum(pvalues_mat <= t, axis=axis) / 2
        min_idx = np.where(n_changes == np.min(n_changes))[0]
        cpdag = (np.sum(dags[min_idx], axis=0) > 0).astype(int)
        precisions.append(dag_precision(true_dag, cpdag))
        recalls.append(dag_recall(true_dag, cpdag))

        # ap_score += precision * (recall - prior_recall)
        # prior_recall = recall

    # if len(thresholds) == 1:
    #     ap_score = precisions[0] * recalls[0]
    # else:
    sort_idx = np.argsort(recalls)
    recalls = np.asarray(recalls)[sort_idx]
    precisions = np.asarray(precisions)[sort_idx]
    ap_score = (np.diff(recalls, prepend=0) * precisions).sum()

    return ap_score
    
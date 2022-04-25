import pytest
import numpy as np
import networkx as nx
from sparse_shift.datasets import erdos_renyi_dag, \
    connected_erdos_renyi_dag, barabasi_albert_dag


def test_are_dags():
    n = 12
    p = 0.5
    G = erdos_renyi_dag(n, p)
    np.testing.assert_array_equal(G, np.tril(G))

    G = connected_erdos_renyi_dag(n, p)
    np.testing.assert_array_equal(G, np.tril(G))

    G = barabasi_albert_dag(n, p)
    np.testing.assert_array_equal(G, np.tril(G))


def test_connected_er_constant():
    n = 12
    p = 0.5
    m = np.sum(connected_erdos_renyi_dag(n, p))

    for _ in range(3):
        assert np.sum(connected_erdos_renyi_dag(n, p)) == m


def test_ba_degree():
    n = 12
    p = 0.5

    G1 = connected_erdos_renyi_dag(n, p)
    G2 = barabasi_albert_dag(n, p)

    assert np.sum(G1.shape[0]) == np.sum(G2.shape[0])

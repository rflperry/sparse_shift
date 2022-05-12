"""Tests for sparse shifts"""
# Authors: Ronan Perry
# License: MIT

import numpy as np
from hyppo.ksample import MMD
from sparse_shift import KCD
from sparse_shift.independence_tests import invariant_residual_test
from sparse_shift.utils import dags2mechanisms
from causallearn.utils.cit import fisherz, kci


def test_dag_shifts(Xs, dags, test='kci', test_kwargs={}, pairwise=True):
    """
    Tests pairwise mechanism equality across a set of dags

    Parameters
    ----------
    Xs : list of np.ndarray, shape (E, n_e, m)
        List of observations from each environment
    dags : np.ndarray, shape (d, m, m)
        List of adjacency matrices
    test : {'invariant_residuals', 'kcd', 'kci', 'fisherz'}, default='kci'
        Test for equality of distribution
    test_kwargs : dict, optional
        Dictionary of named arguments for the independence test

    Returns
    -------
    np.ndarray, shape (d, m, e, e)
        pvalues for each pairwise test
    """
    E = len(Xs)
    M = dags[0].shape[0]
    mech_dict = dags2mechanisms(dags)
    pvalues_dict = {}
    for m, mechanisms in mech_dict.items():
        pvalues_dict[m] = {}
        for parents in mechanisms:
            if pairwise:
                pvalues = test_mechanism(
                    Xs, m, parents, test, test_kwargs
                )
                pvalues_dict[m][tuple(parents)] = pvalues
            else:
                pvalue = test_pooled_mechanism(
                    Xs, m, parents, test, test_kwargs
                )
                pvalues_dict[m][tuple(parents)] = pvalue

    if pairwise:
        dag_pvalues = np.zeros((len(dags), M, E, E))
    else:
        dag_pvalues = np.zeros((len(dags), M))

    for i, dag in enumerate(dags):
        for m, parents in enumerate(dag.T):  # transpose to get parents
            dag_pvalues[i, m] = pvalues_dict[m][tuple(parents)]

    return dag_pvalues


def test_mechanism_shifts(Xs, dag, test='kci', test_kwargs={}, alpha=0.05, pairwise=True):
    """
    Tests pairwise mechanism equality

    Parameters
    ----------
    Xs : list of np.ndarray, shape (E, n_e, m)
        List of observations from each environment
    dag : np.ndarray, shape (m, m)
        Adjacency matrix
    test : {'invariant_residuals', 'kcd'}, default='kci'
        Test for equality of distribution
    test_kwargs : dict, optional
        Dictionary of named arguments for the independence test

    Returns
    -------
    int : total number of shifts
    np.ndarray, shape (e, e, m)
        pvalues for each pairwise test
    """
    E = len(Xs)
    parent_graph = np.asarray(dag).T
    M = parent_graph.shape[0]

    pvalues = np.ones((M, E, E))

    for m in range(M):
        pvalues[m] = test_mechanism(
            Xs, m, parent_graph[m], test, test_kwargs
            )

    num_shifts = np.sum(pvalues <= alpha) // 2

    return num_shifts, pvalues


def test_pooled_mechanism(Xs, m, parents, test='kci', test_kwargs={}):
    parents = np.asarray(parents).astype(bool)

    if test == 'fisherz':
        assert len(Xs) > 1
        # Test X \indep E | PA_X
        data = np.block([
            [np.reshape([e] * self.Xs_[e].shape[0], (-1, 1)), self.Xs_[e]]
            for e in range(len(self.Xs_))
        ])
        condition_set = tuple(np.where(parents > 0)[0] + 1)
        pvalue = fisherz(data, 0, m+1, condition_set)
    elif test == 'kci':
        assert len(Xs) > 1
        # Test X \indep E | PA_X
        data = np.block([
            [np.reshape([e] * Xs[e].shape[0], (-1, 1)), Xs[e]]
            for e in range(len(Xs))
        ])
        condition_set = tuple(np.where(parents > 0)[0] + 1)
        pvalue = kci(data, 0, m+1, condition_set)
    else:
        raise ValueError(f'Test {test} not implemented.')

    return pvalue

def test_mechanism(Xs, m, parents, test='kci', test_kwargs={}):
    """Tests a mechanism"""

    E = len(Xs)
    parents = np.asarray(parents).astype(bool)
    pvalues = np.ones((E, E))

    for e1 in range(E):
        for e2 in range(e1 + 1, E):
            if sum(parents) == 0:
                stat, pvalue = MMD().test(
                    Xs[e1][:, m].reshape(-1, 1),
                    Xs[e2][:, m].reshape(-1, 1),
                )
            else:
                if test == 'kcd':
                    assert len(Xs) == 2
                    _, pvalue = KCD(n_jobs=test_kwargs['n_jobs']).test(
                        np.vstack((Xs[e1][:, parents], Xs[e2][:, parents])),
                        np.concatenate((Xs[e1][:, m], Xs[e2][:, m])),
                        np.asarray([0] * Xs[e1].shape[0] + [1] * Xs[e2].shape[0]),
                        reps=test_kwargs['n_reps'],
                    )
                elif test == 'invariant_residuals':
                    assert len(Xs) == 2
                    pvalue, *_ = invariant_residual_test(
                        np.vstack((Xs[e1][:, parents], Xs[e2][:, parents])),
                        np.concatenate((Xs[e1][:, m], Xs[e2][:, m])),
                        np.asarray([0] * Xs[e1].shape[0] + [1] * Xs[e2].shape[0]),
                        **test_kwargs
                    )
                elif test == 'fisherz':
                    assert len(Xs) > 1
                    # Test X \indep E | PA_X
                    data = np.block([
                        [np.reshape([0] * Xs[e1].shape[0], (-1, 1)), Xs[e1]],
                        [np.reshape([1] * Xs[e2].shape[0], (-1, 1)), Xs[e2]]
                    ])
                    condition_set = tuple(np.where(parents > 0)[0] + 1)
                    pvalue = fisherz(data, 0, m+1, condition_set)
                elif test == 'kci':
                    assert len(Xs) > 1
                    # Test X \indep E | PA_X
                    data = np.block([
                        [np.reshape([0] * Xs[e1].shape[0], (-1, 1)), Xs[e1]],
                        [np.reshape([1] * Xs[e2].shape[0], (-1, 1)), Xs[e2]]
                    ])
                    condition_set = tuple(np.where(parents > 0)[0] + 1)
                    pvalue = kci(data, 0, m+1, condition_set)
                else:
                    raise ValueError(f'Test {test} not implemented.')
            pvalues[e1, e2] = pvalue
            pvalues[e2, e1] = pvalue

    return pvalues

"""Tools for other functions and methods"""
import numpy as np


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
        for i, mech in enumerate(dag):
            mech_dict[i].append(mech)

    # remove duplicates
    for i in range(m):
        mech_dict[i] = np.unique(mech_dict[i], axis=0)

    return mech_dict

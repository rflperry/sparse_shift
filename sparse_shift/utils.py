"""Tools for other functions and methods"""


def check_2d(X):
    if X is not None and X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

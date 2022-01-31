"""Kernel Conditional Discrepancy test"""

# Author: Ronan Pery

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from joblib import Parallel, delayed
from .utils import check_2d


def _compute_kern(X, Y=None, metric='rbf', n_jobs=None, sigma=None):
    """Computes an RBF kernel matrix using median l2 distance as bandwidth"""
    X = check_2d(X)
    Y = check_2d(Y)
    if sigma is None:
        l2 = pairwise_distances(X, metric="l2", n_jobs=n_jobs)
        n = l2.shape[0]
        # compute median of off diagonal elements
        med = np.median(
            np.lib.stride_tricks.as_strided(
                l2, (n - 1, n + 1), (l2.itemsize * (n + 1), l2.itemsize)
            )[:, 1:]
        )
        # prevents division by zero when used on label vectors
        med = med if med else 1
    else:
        med = sigma
    gamma = 1.0 / (2 * (med ** 2))
    return pairwise_kernels(X, Y=Y, metric=metric, n_jobs=n_jobs, gamma=gamma), med


class KCD():
    """
    Kernel Conditional Discrepancy test.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean" or "gaussian"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``compute_distance`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Alternatively, this function computes the kernel similarity among the
        samples within each data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
    reg : float, nonnegative
        Amount of regularization for inverting the kernel matrix.
    n_jobs : int, optional
        Number of jobs to run computations in parallel.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    Attributes
    ----------
    sigma_x_ : float
        median l2 distance between conditional features X
    sigma_x_ : float
        median l2 distance between target features Y 
    null_dist_ : list
        Null distribution of test statistics after calling test.
    e_hat_ : list
        Propensity score probabilities of samples being in group 1.
    """

    def __init__(self, compute_distance=None, reg=0.001, n_jobs=None, **kwargs):
        self.compute_distance = compute_distance
        self.reg = reg
        self.kwargs = kwargs
        self.n_jobs = n_jobs

    def statistic(self, X, Y, z):
        r"""
        Calulates the 2-sample test statistic.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Features to condition on
        Y : ndarray, shape (n, q)
            Target or outcome features
        z : list or ndarray, length n
            List of zeros and ones indicating which samples belong to
            which groups.

        Returns
        -------
        stat : float
            The computed statistic
        """
        K, sigma_x = _compute_kern(X, n_jobs=self.n_jobs)
        L, sigma_y = _compute_kern(Y, n_jobs=self.n_jobs)
        self.sigma_x_ = sigma_x
        self.sigma_y_ = sigma_y

        return self._statistic(K, L, z)

    def _get_inverse_kernels(self, K, z):
        """Helper function to compute W matrices"""
        # Compute W matrices from z
        K0 = K[np.array(1 - z, dtype=bool)][:, np.array(1 - z, dtype=bool)]
        K1 = K[np.array(z, dtype=bool)][:, np.array(z, dtype=bool)]

        W0 = np.linalg.inv(K0 + self.reg * np.identity(int(np.sum(1 - z))))
        W1 = np.linalg.inv(K1 + self.reg * np.identity(int(np.sum(z))))

        return W0, W1

    def _statistic(self, K, L, z):
        """Helper function for efficient permutation calculations"""
        # Compute W matrices from z
        W0, W1 = self._get_inverse_kernels(K, z)

        # Compute L kernels
        L0 = L[np.array(1 - z, dtype=bool)][:, np.array(1 - z, dtype=bool)]
        L1 = L[np.array(z, dtype=bool)][:, np.array(z, dtype=bool)]
        L01 = L[np.array(1 - z, dtype=bool)][:, np.array(z, dtype=bool)]

        # Compute test statistic using traces
        # Simplified to avoid repeat computations. W symmetric
        KW0 = K[:, np.array(1 - z, dtype=bool)] @ W0
        KW1 = K[:, np.array(z, dtype=bool)] @ W1
        first = np.trace(KW0.T @ KW0 @ L0)
        second = np.trace(KW1.T @ KW0 @ L01)
        third = np.trace(KW1.T @ KW1 @ L1)

        return (first - 2 * second + third) / K.shape[0]

    def witness(self, X, Y, z, X_wit, Y_wit):
        r"""
        Calulates the witness function on a set of points

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Features to condition on
        Y : ndarray, shape (n, q)
            Target or outcome features
        z : list or ndarray, length n
            List of zeros and ones indicating which samples belong to
            which groups.
        X_wit : ndarray, shape (m, p)
            Features to compute the witness distance to
        Y_wit : ndarray, shape (l, q)
            Target or outcome features for the witness distance

        Returns
        -------
        dists : ndarray, shape (l, m)
            The computed distances for all X_wit, Y_wit points
        """
        K, sigma_x = _compute_kern(X, n_jobs=self.n_jobs)
        self.sigma_x_ = sigma_x

        # Compute W matrices from z
        W0, W1 = self._get_inverse_kernels(K, z)
        del K

        # Witness distances
        K, _ = _compute_kern(X, X_wit, n_jobs=self.n_jobs, sigma=sigma_x)
        L, sigma_y = _compute_kern(Y, Y_wit, n_jobs=self.n_jobs)
        self.sigma_y_ = sigma_y

        K0 = K[np.array(1 - z, dtype=bool)]
        K1 = K[np.array(z, dtype=bool)]
        L0 = L[np.array(1 - z, dtype=bool)]
        L1 = L[np.array(z, dtype=bool)]

        return (K1.T @ W1 @ L1 - K0.T @ W0 @ L0).T

    # def conditional_dmat(self, X, Y):
    #     """Pairwise distances between conditional features, w.r.t. conditional dist"""
    #     K, _ = _compute_kern(X, n_jobs=self.n_jobs)
    #     L, _ = _compute_kern(Y, n_jobs=self.n_jobs)

    #     W = np.linalg.inv(K + self.reg * np.identity(K.shape[0]))

    #     XY = K.T @ W @ L
    #     return pairwise_distances(XY, metric="l2", n_jobs=self.n_jobs)

    # def U_regress(self, X, Y, X_predict, alpha=1):
    #     """
    #     Generalized kernel ridge regression for U-statistic regression at X_predict
    #     """
    #     X = check_2d(X)
    #     Y = check_2d(Y)
    #     X_predict = check_2d(X_predict)
    #     X_predict = np.tile(X_predict, 2)
    #     n = X.shape[0]
    #     p = X.shape[1]
    #     x_pairs = np.zeros((n ** 2, 2 * p))
    #     x_pairs[:, range(p)] = np.repeat(X, n, axis=0)
    #     x_pairs[:, range(p, 2 * p)] = np.tile(X, (n, 1))
    #     h = np.reshape(0.5 * (np.power(Y, 2) + np.power(Y, 2).T -
    #                           2 * np.matmul(Y, Y.T)), -1)

    #     var = KernelRidge(
    #         alpha=alpha, kernel='rbf', gamma=1.0 / (2 * (self.sigma_x_ ** 2))
    #     ).fit(x_pairs, h).predict(X_predict)

    #     return var

    def test(self, X, Y, z, reps=1000, random_state=None):
        r"""
        Calculates the *k*-sample test statistic and p-value.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.

        Returns
        -------
        stat : float
            The computed *k*-sample statistic.
        pvalue : float
            The computed *k*-sample p-value.
        """
        # Construct kernel matrices for X, Y TODO efficient storage
        K, sigma_x = _compute_kern(X, n_jobs=self.n_jobs)
        L, sigma_y = _compute_kern(Y, n_jobs=self.n_jobs)
        self.sigma_x_ = sigma_x
        self.sigma_y_ = sigma_y

        # Compute W matrices from z
        stat = self._statistic(K, L, z)

        # Compute proensity scores
        # Note: for stability should maybe exclude samples w/ prob < 1/reps
        self.e_hat_ = LogisticRegression(
            n_jobs=self.n_jobs, penalty='l2', warm_start=True,
            solver='lbfgs', C=1 / self.reg).fit(
                K, z
        ).predict_proba(K)[:, 1]

        # Parallelization  storage cost of kernel matrices
        self.null_dist_ = np.array(
            Parallel(n_jobs=self.n_jobs)(
                [
                    delayed(self._statistic)(
                        K, L, np.random.binomial(1, self.e_hat_))
                    for _ in range(reps)
                ]
            )
        )
        pvalue = (1 + np.sum(self.null_dist_ >= stat)) / (1 + reps)

        return stat, pvalue

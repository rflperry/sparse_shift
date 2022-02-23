"""Kernel Conditional Discrepancy test"""

# Author: Ronan Pery

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed
from .utils import check_2d


def _compute_kern(X, Y=None, metric="rbf", n_jobs=None, sigma=None):
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


def _compute_reg_bound(K):
    n = K.shape[0]
    evals = np.linalg.svd(K, compute_uv=False, hermitian=True)
    res = minimize_scalar(
        lambda reg: np.sum(evals ** 2 / (evals + reg) ** 2) / n + reg,
        bounds=(0.0001, 1000),
        method="bounded",
    )
    return res.x


class KCD:
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
    regs : (float, float), default=None
        Amount of regularization for inverting the kernel matrices
        of the two classes.
        If None, chooses the value that minimizes the upper bound
        on the mean squared prediction error.
    n_jobs : int, optional
        Number of jobs to run computations in parallel.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    Attributes
    ----------
    sigma_x_ : float
        median l2 distance between conditional features X
    sigma_y_ : float
        median l2 distance between target features Y
    regs_ : (float, float)
        Regularization used to invert kernel matrices in X
    propensity_reg_ : float
        Regularization parameter computed for propensity scores
    null_dist_ : list
        Null distribution of test statistics after calling test.
    e_hat_ : list
        Propensity score probabilities of samples being in group 1.

    Notes
    -----
    Per [1], the regularization level should scale with n_samples**b,
    where b is in (0, 0.5)

    References
    ----------
    [1] J. Park, U. Shalit, B. Schölkopf, and K. Muandet, “Conditional Distributional
        Treatment Effect with Kernel Conditional Mean Embeddings and U-Statistic
        Regression,” arXiv:2102.08208, Jun. 2021.
    """

    def __init__(self, compute_distance=None, regs=None, n_jobs=None, **kwargs):
        self.compute_distance = compute_distance
        self.regs = regs
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

        if not hasattr(self, "regs_"):
            self.regs_ = self.regs
        if self.regs_ is None:
            self.regs_ = (_compute_reg_bound(K0), _compute_reg_bound(K1))

        W0 = np.linalg.inv(K0 + self.regs_[0] * np.identity(int(np.sum(1 - z))))
        W1 = np.linalg.inv(K1 + self.regs_[1] * np.identity(int(np.sum(z))))

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

    def test(self, X, Y, z, reps=1000, random_state=None, fast_pvalue=None):
        r"""
        Calculates the *k*-sample test statistic and p-value.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Features to condition on
        Y : ndarray, shape (n, q)
            Target or outcome features
        z : list or ndarray, length n
            List of zeros and ones indicating which samples belong to
            which groups.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        random_state : int, RandomState instance or None, default=None
            Controls randomness in propensity score estimation.
        fast_pvalue : Ignored
            Ignored

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
        self.propensity_reg_ = _compute_reg_bound(K)
        # Note: for stability should maybe exclude samples w/ prob < 1/reps
        self.e_hat_ = (
            LogisticRegression(
                n_jobs=self.n_jobs,
                penalty="l2",
                warm_start=True,
                solver="lbfgs",
                random_state=random_state,
                C=1 / (2 * self.propensity_reg_),
            )
            .fit(K, z)
            .predict_proba(K)[:, 1]
        )

        # Parallelization  storage cost of kernel matrices
        self.null_dist_ = np.array(
            Parallel(n_jobs=self.n_jobs)(
                [
                    delayed(self._statistic)(K, L, np.random.binomial(1, self.e_hat_))
                    for _ in range(reps)
                ]
            )
        )
        pvalue = (1 + np.sum(self.null_dist_ >= stat)) / (1 + reps)

        return stat, pvalue


class KCDCV:
    """
    Kernel Conditional Discrepancy test with cross validation for hyperparamter
    selection.

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
    regs : list, shape (n_regs,), default=(0.01, 0.1, 1.0, 10, 100)
        List of kernel regularization values to try. Larger values correspond
        to larger regularization.
    n_jobs : int, optional
        Number of jobs to run computations in parallel.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.

    Attributes
    ----------
    sigma_X_ : float
        median l2 distance between training features X
    sigma_Y_ : float
        median l2 distance between all training targets Y
    train_idx_ : numpy.ndarray
        Indices of training subset
    test_idx_ : numpy.ndarray
        Indices of test subset
    reg_snrs_ : list
        Training test statistic SNRs for each regularization value
    reg_opt_ : float
        Regularization value with maximum SNR on the training data
    stat_ : float
        Test data test statistic
    stat_snr_ : float
        Test data SNR
    power_alphas_ : numpy.ndarray
        False positive rate values for power calculation
    analytic_powers_ : numpy.ndarray
        Analytic true positive (power) for varying alpha levels
    analytic_pvalue_ : float
        Analytic pvalue based on a normal distribution approximation
    e_hat_ : list
        Propensity score probabilities, P(Z=1 | X)
    null_stats_ : list
        Null distribution of test statistics on permuted test Y
    null_vars_ : numpy.ndarray
        Estimated variance of null distribution test statistics
    perm_pvalue : float
        Pvalue of the test data statistic based on a permutation test

    References
    ----------
    [1] J. Park, U. Shalit, B. Schölkopf, and K. Muandet, “Conditional
        Distributional Treatment Effect with Kernel Conditional Mean
        Embeddings and U-Statistic Regression,” arXiv:2102.08208, Jun. 2021.
    [2] J. M. Kübler, W. Jitkrittum, B. Schölkopf, and K. Muandet, “A Witness
        Two-Sample Test,” arXiv:2102.05573, Feb. 2022.
    """

    def __init__(
        self,
        compute_distance=None,
        regs=(0.01, 0.1, 1.0, 10, 100),
        n_jobs=None,
        **kwargs
    ):
        self.compute_distance = compute_distance
        self.regs = regs
        self.kwargs = kwargs
        self.n_jobs = n_jobs

    def _optimize_params(self, X, Y, z, test_fraction=None, random_state=None):
        """
        Optimizes the regularization amount for the kernel matrix inversion
        as detailed in [2]. Splits the data based on labels z and computes
        the witness function using the training data. The optimal amount of
        regularization is that which maximizes the signal to noise ratio of
        the difference between the two distributions.
        """
        X = check_2d(X)
        Y = check_2d(Y)
        self.n_samples_, self.n_features_ = X.shape

        # Split data into train/test
        self.train_idx_, self.test_idx_ = next(
            StratifiedShuffleSplit(
                n_splits=1,
                test_size=0.5 if test_fraction is None else test_fraction,
                random_state=random_state,
            ).split(np.zeros(self.n_samples_), z)
        )
        self.train_idx_ = np.sort(self.train_idx_)
        self.test_idx_ = np.sort(self.test_idx_)

        self.train_idx_ = np.arange(self.n_samples_)
        self.test_idx_ = np.arange(self.test_idx_)

        # compute K0, K1 on all data. split so as to compute separate sigmas
        _, self.sigma_X_ = _compute_kern(X[self.train_idx_], n_jobs=self.n_jobs)

        K00, _ = _compute_kern(
            X[self.train_idx_][np.array(1 - z[self.train_idx_], dtype=bool)],
            n_jobs=self.n_jobs,
            sigma=self.sigma_X_,
        )
        K01, _ = _compute_kern(
            X[self.train_idx_][np.array(1 - z[self.train_idx_], dtype=bool)],
            X[self.train_idx_][np.array(z[self.train_idx_], dtype=bool)],
            sigma=self.sigma_X_,
            n_jobs=self.n_jobs,
        )
        K0 = np.hstack((K00, K01))

        K11, _ = _compute_kern(
            X[self.train_idx_][np.array(z[self.train_idx_], dtype=bool)],
            n_jobs=self.n_jobs,
            sigma=self.sigma_X_,
        )
        K10, _ = _compute_kern(
            X[self.train_idx_][np.array(z[self.train_idx_], dtype=bool)],
            X[self.train_idx_][np.array(1 - z[self.train_idx_], dtype=bool)],
            sigma=self.sigma_X_,
            n_jobs=self.n_jobs,
        )
        K1 = np.hstack((K11, K10))

        # compute l0, l1 on training data
        _, self.sigma_Y_ = _compute_kern(
            Y[self.train_idx_],  # [np.array(1 - z[self.train_idx_], dtype=bool)],
            n_jobs=self.n_jobs,
        )
        L0, _ = _compute_kern(
            Y[self.train_idx_][np.array(1 - z[self.train_idx_], dtype=bool)],
            Y[self.train_idx_],  # [np.array(1 - z[self.train_idx_], dtype=bool)],
            n_jobs=self.n_jobs,
            sigma=self.sigma_Y_,
        )

        L1, _ = _compute_kern(
            Y[self.train_idx_][np.array(z[self.train_idx_], dtype=bool)],
            Y[self.train_idx_],  # [np.array(z[self.train_idx_], dtype=bool)],
            n_jobs=self.n_jobs,
            sigma=self.sigma_Y_,
        )

        # indices of groups 0 and 1 in L0, L1
        idx = z[self.train_idx_]

        # Iterate over reg
        self.reg_snrs_ = []
        for reg in self.regs:  # could consider separate reg parameters
            W0 = np.linalg.inv(K00 + reg * np.identity(K00.shape[0]))
            K0W0 = K0.T @ W0
            W1 = np.linalg.inv(K11 + reg * np.identity(K11.shape[0]))
            K1W1 = K1.T @ W1

            stat, pooled_var = self._statistic(K0W0, L0, K1W1, L1, idx)

            self.reg_snrs_.append(stat / np.sqrt(pooled_var))

        # identify optimal reg
        self.reg_opt_ = self.regs[np.argmax(np.abs(self.reg_snrs_))]
        h_sign = np.sign(self.reg_snrs_[np.argmax(np.abs(self.reg_snrs_))])

        W0 = np.linalg.inv(K00 + self.reg_opt_ * np.identity(K00.shape[0]))
        W1 = np.linalg.inv(K11 + self.reg_opt_ * np.identity(K11.shape[0]))

        return W0, W1, h_sign

    def test(self, X, Y, z, reps=1000, random_state=None, fast_pvalue=False):
        r"""
        Calculates the *k*-sample test statistic and p-value using the optimal
        regularization value which maximizes the test statistic SNR.

        Parameters
        ----------
        X : ndarray, shape (n, p)
            Features to condition on
        Y : ndarray, shape (n, q)
            Target or outcome features
        z : list or ndarray, length n
            List of zeros and ones indicating which samples belong to
            which groups.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        random_state : int, RandomState instance or None, default=None
            Controls the randomness of the data splitting.
        fast_pvalue : boolean, default=False
            If True, the analytic form of the pvalue is computed using a
            normal distribution approximation. Valid with larger sample sizes.

        Returns
        -------
        stat : float
            The computed test statistic.
        pvalue : float
            The computed test p-value.
        """
        W0, W1, h_sign = self._optimize_params(X, Y, z, random_state=random_state)

        K, _ = _compute_kern(X, sigma=self.sigma_X_, n_jobs=self.n_jobs,)

        K0W0 = (
            K[self.train_idx_][np.array(1 - z[self.train_idx_], dtype=bool)][
                :, self.test_idx_
            ].T
            @ W0
        )
        K1W1 = (
            K[self.train_idx_][np.array(z[self.train_idx_], dtype=bool)][
                :, self.test_idx_
            ].T
            @ W1
        )

        # compute l0, l1 from test to train
        L0, _ = _compute_kern(  # shape (tr0, te)
            Y[self.train_idx_][np.array(1 - z[self.train_idx_], dtype=bool)],
            Y[self.test_idx_],
            sigma=self.sigma_Y_,
            n_jobs=self.n_jobs,
        )
        L1, _ = _compute_kern(  # shape (tr1, te)
            Y[self.train_idx_][np.array(z[self.train_idx_], dtype=bool)],
            Y[self.test_idx_],
            sigma=self.sigma_Y_,
            n_jobs=self.n_jobs,
        )

        # binary indices for test stat
        idx = z[self.test_idx_]
        n0 = np.array(1 - z[self.test_idx_], dtype=bool).sum()
        n1 = np.array(z[self.test_idx_], dtype=bool).sum()

        # h_sign ensures learned kernel expectation for Y1 > Y0
        self.stat_, pooled_var = self._statistic(K0W0, L0, K1W1, L1, idx, sign=h_sign)
        self.stat_snr_ = self.stat_ / np.sqrt(pooled_var)

        # Compute analytic power, per [2]
        self.power_alphas_ = np.linspace(0.05, 1, 21)
        self.analytic_powers_ = 1 - np.asarray(
            [
                norm.cdf(norm.ppf(1 - alpha) - self.stat_snr_ * np.sqrt(n0 + n1))
                for alpha in self.power_alphas_
            ]
        )

        # permutation test via permute l0, l1 per propensity scores
        # Note: for stability should maybe exclude samples w/ prob < 1/reps
        # Is trained on entire dataset, but then subsetted in the test step
        if fast_pvalue:  # per [2]
            self.analytic_pvalue_ = 1 - norm.cdf(self.stat_snr_ * np.sqrt(n0 + n1))
            return self.stat_, self.analytic_pvalue_
        else:
            self.e_hat_ = (
                LogisticRegression(
                    n_jobs=self.n_jobs,
                    penalty="l2",
                    warm_start=True,
                    solver="lbfgs",
                    C=1 / (2 * self.reg_opt_),
                    random_state=random_state,
                )
                .fit(K, z)
                .predict_proba(K)[:, 1]
            )

            M00 = K0W0.T @ K0W0
            M01 = K0W0.T @ K1W1
            M11 = K1W1.T @ K1W1

            h0_list = M00 @ L0 - M01 @ L1
            h1_list = M01.T @ L0 - M11 @ L1
            # Parallelization, storage cost of kernel matrices
            self.null_stats_, self.null_vars_ = np.array(
                list(
                    zip(
                        *Parallel(n_jobs=self.n_jobs)(
                            [
                                delayed(self._permute_statistic)(
                                    h0_list,
                                    h1_list,
                                    # K0W0, L0, K1W1, L1,
                                    np.random.binomial(1, self.e_hat_[self.test_idx_]),
                                    sign=h_sign,
                                )
                                for _ in range(reps)
                            ]
                        )
                    )
                )
            )
            self.perm_pvalue_ = (1 + np.sum(self.null_stats_ >= self.stat_)) / (
                1 + reps
            )

            return self.stat_, self.perm_pvalue_

    def _permute_statistic(self, h0_list, h1_list, idx, sign=1):
        h0_list = h0_list[:, np.array(1 - idx, dtype=bool)]
        h1_list = h1_list[:, np.array(idx, dtype=bool)]

        c = len(h0_list) / (len(h0_list) + len(h1_list))
        pooled_var = np.var(h0_list) / c + np.var(h1_list) / (1 - c)
        stat = (np.mean(h1_list) - np.mean(h0_list)) * sign
        return stat, pooled_var

    def _statistic(self, K0W0, L0, K1W1, L1, idx, sign=1):
        # test statistic and variance
        idx0 = np.array(1 - idx, dtype=bool)
        idx1 = np.array(idx, dtype=bool)

        M00 = K0W0.T @ K0W0
        M01 = K0W0.T @ K1W1
        M11 = K1W1.T @ K1W1

        h0_list = M00 @ L0[:, idx0] - M01 @ L1[:, idx0]
        h1_list = M01.T @ L0[:, idx1] - M11 @ L1[:, idx1]

        c = len(h0_list) / (len(h0_list) + len(h1_list))
        pooled_var = np.var(h0_list) / c + np.var(h1_list) / (1 - c)
        stat = (np.mean(h1_list) - np.mean(h0_list)) * sign
        return stat, pooled_var

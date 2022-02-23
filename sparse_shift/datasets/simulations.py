"""Simulated causal datasets"""

import numpy as np
from functools import partial


def sample_topological(n, equations, noise, random_state=None):
    """
    Samples from a Structural Causal Model (SCM) in topological order

    Parameters
    ----------
    n : int
        Number of observations to sample

    equations : list of callables
        List of SCM equations, each a function accepting two parameters
        - All variables, of course only parents will be used.
        - An exogenous noise variable.

    noise : callable or list of callables
        Exogenous noise for each structural equation. If a single callable,
        then the same function will be used for all equations.

    random_state : int, optional
        Seed for reproducible randomness.

    Returns
    -------
    np.ndarray, shape (n, len(equations))
        Sampled observational data
    """
    np.random.seed(random_state)
    n_vars = len(equations)
    X = np.zeros((n_vars, n))

    if not callable(noise):
        assert len(equations) == len(
            noise
        ), f"Must provide the same number of structural \
            equations as noise variables. Provided {len(equations)} and \
                {len(noise)}"

    for i, f in enumerate(equations):
        if not callable(noise):
            u = np.asarray([noise[i]() for _ in range(n)])
        else:
            u = np.asarray([noise() for _ in range(n)])
        X[i] = f(X, u)

    return X.T


def _icp_base_func(X, u, parents, function, f_join):
    """Helper function for icp simulations"""
    X = X * parents
    X = X[parents != 0]
    return f_join(function(X)) + u


def nonlinear_icp_sim(
    n_samples,
    dag,
    function=lambda x: x,
    noise_df=2,
    function_conjunction="additive",
    intervention="soft",
    intervention_shift=0,
    intervention_scale=0,
    intervention_pct=0,
    random_state=None,
):
    """
    Simulates data from a given dag according to the simulation design
    in Heinz-Deml et al. 2018

    """
    noise = lambda: np.random.standard_t(df=noise_df)
    if function_conjunction == "additive":
        f_join = np.sum
    elif function_conjunction == "multiplicative":
        f_join = np.prod

    equations = [
        partial(_icp_base_func, parents=parents, function=function, f_join=f_join,)
        for parents in dag
    ]

    if intervention_pct > 0:
        n_intervene = np.ceil(len(dag) * intervention_pct)
        intervene_idx = np.random.choice(len(dag), n_intervene, replace=False)
        for i in intervene_idx:
            if intervention == "soft":
                pass
                # equations[i] = lambda X, U:
            elif intervention == "hard":
                pass
            else:
                raise ValueError(
                    f"Intervention value {intervention} \
                    not accepted"
                )

    X = sample_topological(n_samples, equations, noise, random_state,)

    return X

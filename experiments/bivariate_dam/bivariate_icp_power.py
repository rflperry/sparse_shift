import numpy as np
from sparse_shift.datasets import sample_nonlinear_icp_sim
from sparse_shift.testing import test_mechanism_shifts
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


PARAMS_DICT = {
    'nonlinearity': ['sqrt'],
    'noise_df': [100],
    'intervention': [
        'soft',
        # lambda X: np.sin(2*np.pi*X)
    ],
    'combination': ['additive'],
    'intervention_shift': [2],
    'intervention_scale': [0],
    'pre_intervention': [False],
    'intervention_targets': [
        [None, 0],
        [None, 1],
    ]
}

N_SAMPLES = [50, 100, 200, 300]

N_REPS = 40

# Parents
dag_dict = {
    'XY': np.asarray([[0, 1], [0, 0]]),
    'YX': np.asarray([[0, 0], [1, 0]]),
}

TRUE_PARENTS = dag_dict['YX']

n_features = list(dag_dict.values())[0].shape[0]
n_dags = len(dag_dict.keys())

alpha = 0.05 / n_features

param_keys, param_values = zip(*PARAMS_DICT.items())
params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

dag_keys, dags = zip(*dag_dict.items())

# Create results csv header
f_name = './bivariate_icp_pvalues.csv'
header0 = np.hstack([
    ['Params'] * (len(param_keys) + 3),
    np.hstack([[key] * n_features for key in dag_keys])
])
header1 = np.hstack([
    ['params_index'], list(param_keys), ['n_samples'], ['rep'],
    np.hstack([[f'X{i+1}' for i in range(n_features)] for _ in range(n_dags)])
])
f = open(f_name, 'w+')
f.write(", ".join(header0) + "\n")
f.write(", ".join(header1) + "\n")
f.flush()

for i, params in enumerate(params_grid):
    intervention_targets = params.pop('intervention_targets')
    for n_samples in N_SAMPLES:
        for rep in range(N_REPS):
            Xs = [
                sample_nonlinear_icp_sim(
                    TRUE_PARENTS,
                    n_samples=n_samples,
                    intervention_targets=targets,
                    random_state=len(intervention_targets)*rep+j,
                    **params
                )
                for j, targets in enumerate(intervention_targets)
            ]
            row = [i] + list(params.values()) + [str(intervention_targets).replace(', ', ','), n_samples, rep]

            if rep == 0 and n_samples == max(N_SAMPLES):
                df = pd.DataFrame(
                    np.hstack((
                        np.vstack(Xs),
                        np.hstack([[i]*X.shape[0] for i, X in enumerate(Xs)]).reshape(-1, 1).astype(int)
                    )),
                    columns=[f'X{i+1}' for i in range(Xs[0].shape[1])] + ['y']
                )
                sns.pairplot(df, hue='y', diag_kind='kde')
                plt.suptitle(f'Param set {i}')
                plt.savefig(f'./figures/bivariate_icp_data_params={i}.pdf')

            for dag in dags:
                num_shifts, pvalues_mat = test_mechanism_shifts(Xs, dag, reps=100, n_jobs=-2)
                n_changes = np.sum(pvalues_mat <= alpha) // 2  # Number of changing mechanisms in DAG
                pvalues = pvalues_mat[:, 0, 1]
                row += list(np.round(pvalues, 3))
            f.write(", ".join(map(str, row)) + "\n")
            f.flush()

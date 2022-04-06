import numpy as np
from sparse_shift.datasets import sample_nonlinear_icp_sim
from sparse_shift.testing import test_mechanism_shifts
import pandas as pd
import pickle
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


PARAMS_DICT = {
    'nonlinearity': ['id'],
    'noise_df': [10],
    'n_samples': [100],
    'intervention': ['soft'],
    'combination': ['additive'],  # 
    'intervention_shift': [2],
    'intervention_scale': [1],
    'intervention_targets': [
        [None, 0],
        [None, 1],
        [None, 2],
        [None, (0, 1)],
        [None, (1, 2)],
        [None, (0, 2)],
    ]
}

N_REPS = 20

TRUE_PARENTS = np.asarray([[0,  0, 0], [1, 0, 0], [1, 1, 0]])

with open("./dag_dict_all_triangles.pkl", "rb") as f:
    dag_dict = pickle.load(f)

# Restrict to MEC
dag_dict = {
    key: dag for key, dag in dag_dict.items() if np.sum(dag) == 3
}

n_features = list(dag_dict.values())[0].shape[0]
n_dags = len(dag_dict.keys())

alpha = 0.05 / n_features

param_keys, param_values = zip(*PARAMS_DICT.items())
params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

dag_keys, dags = zip(*dag_dict.items())

# Create results csv header
header0 = np.hstack([
    ['Params'] * (len(param_keys) + 2),
    np.hstack([[key] * 3 for key in dag_keys])
])
header1 = np.hstack([
    ['params_index'], list(param_keys), ['rep'],
    np.hstack([[f'X{i+1}' for i in range(n_features)] for _ in range(n_dags)])
])
f = open('./icp_triangle_changes.csv', 'w+')
f.write(", ".join(header0) + "\n")
f.write(", ".join(header1) + "\n")
f.flush()

for i, params in enumerate(params_grid):
    intervention_targets = params.pop('intervention_targets')
    for rep in range(N_REPS):
        Xs = [
            sample_nonlinear_icp_sim(
                TRUE_PARENTS,
                intervention_targets=targets,
                random_state=len(intervention_targets)*rep+j,
                **params
            )
            for j, targets in enumerate(intervention_targets)
        ]
        row = [i] + list(params.values()) + [str(intervention_targets).replace(', ', ','), rep]

        if rep == 0:
            df = pd.DataFrame(
                np.hstack((
                    np.vstack(Xs),
                    np.hstack([[i]*X.shape[0] for i, X in enumerate(Xs)]).reshape(-1, 1).astype(int)
                )),
                columns=[f'X{i+1}' for i in range(Xs[0].shape[1])] + ['y']
            )
            sns.pairplot(df, hue='y', diag_kind='kde')
            plt.savefig(f'./figures/triangle_icp_data_params={i}.pdf')

        for dag in dags:
            num_shifts, pvalues_mat = test_mechanism_shifts(Xs, dag, reps=100, n_jobs=-2)
            n_changes = np.sum(pvalues_mat <= alpha) // 2  # Number of changing mechanisms in DAG
            pvalues = pvalues_mat[:, 0, 1]
            row += list(np.round(pvalues, 3))
        f.write(", ".join(map(str, row)) + "\n")
        f.flush()

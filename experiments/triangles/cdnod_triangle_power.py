import numpy as np
from sparse_shift.datasets import sample_cdnod_sim
from sparse_shift.testing import test_mechanism_shifts, test_dag_shifts
import pandas as pd
import pickle
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

SIM_NAME = 'cdnod_triangle'
PARAMS_DICT = {
    'intervention_targets': [
        [None, None],
        [None, 0],
        [None, 1],
        [None, 2],
        [None, (0, 1)],
        [None, (1, 2)],
        [None, (0, 2)],
        [None, (1, 2, 3)],
    ]
}

N_REPS = 20

N_SAMPLES = [50, 100, 200]

TRUE_PARENTS = np.asarray([[0,  0, 0], [1, 0, 0], [1, 1, 0]])

with open("./dag_dict_all_triangles.pkl", "rb") as f:
    dag_dict = pickle.load(f)

# Restrict to MEC
dag_dict = {
    key: dag for key, dag in dag_dict.items() if np.sum(dag) == 3
}

n_features = list(dag_dict.values())[0].shape[0]
n_dags = len(dag_dict.keys())

param_keys, param_values = zip(*PARAMS_DICT.items())
params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

dag_keys, dags = zip(*dag_dict.items())

# Create results csv header
f_name = f'./{SIM_NAME}_pvalues.csv'
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
        for rep in tqdm(range(N_REPS)):
            Xs = [
                sample_cdnod_sim(
                    TRUE_PARENTS,
                    n_samples=n_samples,
                    intervention_targets=targets,
                    base_random_state=len(intervention_targets)*rep,
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
                plt.savefig(f'./figures/{SIM_NAME}_data_params={i}.pdf')

            pvalues_mat = test_dag_shifts(Xs, dags, reps=100, n_jobs=-2)
            pvalues = pvalues_mat[:, :, 0, 1]
            for dag_pvalues in pvalues:
                row += list(np.round(dag_pvalues, 3))
            f.write(", ".join(map(str, row)) + "\n")
            f.flush()

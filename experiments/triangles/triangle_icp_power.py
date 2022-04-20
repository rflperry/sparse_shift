import numpy as np
from sparse_shift.datasets import sample_nonlinear_icp_sim
from sparse_shift.testing import test_dag_shifts
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import warnings
import os

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

TEST_GRID = [
    # {
    #     'test': 'invariant_residuals',
    #     'test_kwargs': {'method': 'gam', 'test': 'ks'}
    # },
    # {
    #     'test': 'invariant_residuals',
    #     'test_kwargs': {'method': 'linear', 'test': 'ks'}
    # },
    {'test': 'kci', 'test_kwargs': {}},
    {'test': 'fisherz', 'test_kwargs': {}}
]

PARAMS_DICT = {
    # 'nonlinearity': ['id', 'sqrt', 'sin'],
    # 'noise_df': [10, 100],
    # 'intervention': [
    #     'soft',
    # ],
    # 'combination': ['additive', 'multiplicative'],
    # 'intervention_shift': [0, 1, 2],
    # 'intervention_scale': [0, 1, 2],
    'nonlinearity': ['sqrt'],
    'noise_df': [100],
    'intervention': [
        'soft',
    ],
    'combination': ['additive'],
    'intervention_shift': [2],
    'intervention_scale': [1],
    # 'pre_intervention': [False],
    # Needs to be last in order to preserve the csv naming
    'intervention_targets': [
        [None, 0],
        [None, 1],
        [None, 2],
        [None, (0, 1)],
        [None, (1, 2)],
        [None, (0, 2)],
    ]
}

N_SAMPLES = [50, 100, 200]

N_REPS = 10

# Parents
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
f_name = './triangle_icp_pvalues.csv'
header = np.hstack([
    ['params_index'], list(TEST_GRID[0].keys()), list(param_keys), ['n_samples', 'rep', 'dag'],
    [f'X{i+1}' for i in range(n_features)]
])
f = open(f_name, 'w+')
f.write(", ".join(header) + "\n")
f.flush()

print(f'{len(params_grid)} total iterations')
for i, params in tqdm(enumerate(params_grid)):
    if params['intervention_scale'] == 0 and params['intervention_shift'] == 0:
        continue
    intervention_targets = params.pop('intervention_targets')
    for test_params in TEST_GRID:
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
                row = (
                    [i] +
                    [str(val).replace(', ', ',') for val in test_params.values()] +
                    list(params.values()) +
                    [str(intervention_targets).replace(', ', ','), n_samples, rep]
                )

                # if rep == 0 and n_samples == max(N_SAMPLES):
                #     df = pd.DataFrame(
                #         np.hstack((
                #             np.vstack(Xs),
                #             np.hstack([[i]*X.shape[0] for i, X in enumerate(Xs)]).reshape(-1, 1).astype(int)
                #         )),
                #         columns=[f'X{i+1}' for i in range(Xs[0].shape[1])] + ['y']
                #     )
                #     sns.pairplot(df, hue='y', diag_kind='kde')
                #     plt.suptitle(f'Param set {i}')
                #     plt.savefig(f'./figures/triangle_icp_data_params={i}.pdf')

                dags_pvalues = test_dag_shifts(
                    Xs, dags, **test_params
                )
                for dag_key, pvalues in zip(dag_keys, dags_pvalues):
                    f.write(", ".join(map(
                        str,
                        row + [dag_key] + list(np.round(pvalues[:, 0, 1], 3))
                    )) + "\n")
                    f.flush()

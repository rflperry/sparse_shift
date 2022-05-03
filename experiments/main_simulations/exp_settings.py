from sparse_shift.methods import MinChange

PARAMS_DICT = {
    "DEBUG": {
        "n_variables": [3],
        "n_total_environments": [3],
        "sparsity": [2],
        'intervention_targets': [None],
        "sample_size": [50],
        "dag_density": [0.3],
        "reps": [3],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    },
    "environment_convergence": {
        "n_variables": [6],
        "n_total_environments": [15],
        "sparsity": [1, 2, 4],
        'intervention_targets': [None],
        "sample_size": [500],
        "dag_density": [0.3],
        "reps": [10],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er", "ba"],
    },
    "soft_samples": {
        "n_variables": [6],
        "n_total_environments": [5],
        "sparsity": [1, 2, 4],
        'intervention_targets': [None],
        "sample_size": [50, 100, 200, 300, 500],
        "dag_density": [0.3],
        "reps": [20],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    },
    "oracle_rates": {
        "n_variables": [4, 6, 10],
        "n_total_environments": [2, 5],
        "sparsity": [1, 2, 3, 5, 8],
        'intervention_targets': [None],
        "sample_size": [None],
        "dag_density": [0.1, 0.3, 0.5, 0.7, 0.9],
        "reps": [20],
        "data_simulator": [None],
        "dag_simulator": ["er"],
    },
    "bivariate_power": {
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[0], [1]],
        ],
        "sample_size": [200],
        "dag_density": [None],
        "reps": [20],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }
}


# save name, method name, algo, hpyerparams
ALL_METHODS = [
    (
        'mch_kci',
        'Min changes (kci)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {
                "KernelX": "GaussianKernel",
                "KernelY": "GaussianKernel",
                "KernelZ": "GaussianKernel",
            },
        }
    ),
    (
        'mch_lin',
        'Min changes (linear)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'invariant_residuals',
            'test_kwargs': {'method': 'linear', 'test': "whitney_levene"},
        }
    ),
]

METHODS_DICT = {
    "DEBUG": ALL_METHODS[:1],
    "environment_convergence": ALL_METHODS,
    "soft_samples": ALL_METHODS,
    "oracle_rates": [],
    "bivariate_power": [
        (
            'mch_kci',
            'KCI',
            MinChange,
            {
                'alpha': 0.05,
                'scale_alpha': True,
                'test': 'kci',
                'test_kwargs': {
                    "KernelX": "GaussianKernel",
                    "KernelY": "GaussianKernel",
                    "KernelZ": "GaussianKernel",
                },
            }
        ),
        (
            'mch_lin',
            'Linear',
            MinChange,
            {
                'alpha': 0.05,
                'scale_alpha': True,
                'test': 'invariant_residuals',
                'test_kwargs': {'method': 'linear', 'test': "whitney_levene"},
            }
        ),
        (
            'mch_gam',
            'GAM',
            MinChange,
            {
                'alpha': 0.05,
                'scale_alpha': True,
                'test': 'invariant_residuals',
                'test_kwargs': {'method': 'gam', 'test': "whitney_levene"},
            }
        ),
        (
            'mch_fisherz',
            'FisherZ',
            MinChange,
            {
                'alpha': 0.05,
                'scale_alpha': True,
                'test': 'fisherz',
                'test_kwargs': {},
            }
        ),
        (
            'mch_kcd',
            'KCD',
            MinChange,
            {
                'alpha': 0.05,
                'scale_alpha': True,
                'test': 'fisherz',
                'test_kwargs': {'n_jobs': -2, 'n_reps': 100},
            }
        ),
    ],
}


def get_experiment_params(exp):
    return PARAMS_DICT[exp]


def get_experiments():
    return list(PARAMS_DICT.keys())


def get_experiment_methods(exp):
    return METHODS_DICT[exp]

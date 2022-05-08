from sparse_shift.methods import MinChange

PARAMS_DICT = {
    "DEBUG": [{
        "n_variables": [3],
        "n_total_environments": [3],
        "sparsity": [2],
        'intervention_targets': [None],
        "sample_size": [50],
        "dag_density": [0.3],
        "reps": [3],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    }],
    "pairwise_power": [
        {
            "n_variables": [6],
            "n_total_environments": [15],
            "sparsity": [2],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [50],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [1, 2, 3, 4, 5],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [50],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [50],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [3, 6, 9, 12],
            "n_total_environments": [5],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [500],
            "dag_density": [0.3],
            "reps": [50],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [2],
            'intervention_targets': [None],
            "sample_size": [50, 100, 250, 500, 1000],
            "dag_density": [0.3],
            "reps": [50],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "environment_convergence": [{
        "n_variables": [6],
        "n_total_environments": [10],
        "sparsity": [1, 2, 4],
        'intervention_targets': [None],
        "sample_size": [500],
        "dag_density": [0.3],
        "reps": [20],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    }],
    "soft_samples": [{
        "n_variables": [6],
        "n_total_environments": [5],
        "sparsity": [1, 2, 3, 4, 5, 6],
        'intervention_targets': [None],
        "sample_size": [50, 100, 200, 300, 500],
        "dag_density": [0.3],
        "reps": [20],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    }],
    "oracle_rates": [{
        "n_variables": [6, 8, 10, 12],
        "n_total_environments": [5],
        "sparsity": [1, 3, 5, 7, 9, 11],
        'intervention_targets': [None],
        "sample_size": [None],
        "dag_density": [0.1, 0.3, 0.5, 0.7, 0.9],
        "reps": [20],
        "data_simulator": [None],
        "dag_simulator": ["er"],
    }],
    "oracle_select_rates": [
        {
            "n_variables": [6, 8, 10, 12],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [1, 2, 3, 4, 5, 6, 7],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [15],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3, 0.5, 0.7, 0.9, 0.1],
            "reps": [20],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
    ],
    "bivariate_power": [{
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[], [0, 1]],
        ],
        "sample_size": [200],
        "dag_density": [None],
        "reps": [20],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }]
}


# save name, method name, algo, hpyerparams
ALL_METHODS = [
    (
        'mch_kci',
        'Min changes (KCI)',
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
        'Min changes (Linear)',
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
        'Min changes (GAM)',
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
        'Min changes (FisherZ)',
        MinChange,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'fisherz',
            'test_kwargs': {},
        }
    ),
]

METHODS_DICT = {
    "DEBUG": ALL_METHODS[:1],
    "pairwise_power": ALL_METHODS,
    "environment_convergence": ALL_METHODS,
    "soft_samples": ALL_METHODS[:2],
    "oracle_rates": [],
    "oracle_select_rates": [],
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


def get_param_keys(exp):
    return list(PARAMS_DICT[exp][0].keys())


def get_experiments():
    return list(PARAMS_DICT.keys())


def get_experiment_methods(exp):
    return METHODS_DICT[exp]

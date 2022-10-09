from sparse_shift.methods import MinChange, AugmentedPC, FullMinChanges, ParamChanges

PARAMS_DICT = {
    "pairwise_power": [
        {
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1, 3, 5],
            'intervention_targets': [None],
            "sample_size": [100],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [6],
            "n_total_environments": [5],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [100],
            "dag_density": [0.3, 0.5, 0.7],
            "reps": [2],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [3, 6, 9],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [100],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [50, 100],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
        {
            "n_variables": [6],
            "n_total_environments": [3],
            "sparsity": [1/3],
            'intervention_targets': [None],
            "sample_size": [100],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": ['cdnod'],
            "dag_simulator": ["er"],
        },
    ],
    "oracle_rates": [{
        "n_variables": [4, 6, 8],
        "n_total_environments": [5],
        "sparsity": [1/5, 1/2, 4/5],
        'intervention_targets': [None],
        "sample_size": [None],
        "dag_density": [0.1, 0.5, 0.9],
        "reps": [2],
        "data_simulator": [None],
        "dag_simulator": ["er"],
    }],
    "oracle_select_rates": [
        {
            "n_variables": [6, 8, 10],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [1, 3, 5, 7],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.3],
            "reps": [2],
            "data_simulator": [None],
            "dag_simulator": ["er", 'ba'],
        },
        {
            # Since 'ba' can't handle all of the same settings as 'er'
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.5, 0.9],
            "reps": [2],
            "data_simulator": [None],
            "dag_simulator": ['ba'],
        },
        {
            "n_variables": [8],
            "n_total_environments": [5],
            "sparsity": [0.5],
            'intervention_targets': [None],
            "sample_size": [None],
            "dag_density": [0.5, 0.9, 0.1],
            "reps": [2],
            "data_simulator": [None],
            "dag_simulator": ["er"],
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
        "sample_size": [100],
        "dag_density": [None],
        "reps": [2],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["complete"],
    }],
    "bivariate_multiplic_power": [{
        "n_variables": [2],
        "n_total_environments": [2],
        "sparsity": [None],
        'intervention_targets': [
            [[], [0]],
            [[], [1]],
            [[], []],
            [[], [0, 1]],
        ],
        "sample_size": [100],
        "dag_density": [None],
        "reps": [2],
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
    (
        'full_pc_kci',
        'Full PC (KCI)',
        FullMinChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
            'test': 'kci',
            'test_kwargs': {},
        }
    ),
    (
        'mc',
        'MC',
        ParamChanges,
        {
            'alpha': 0.05,
            'scale_alpha': True,
        }
    )
]

METHODS_DICT = {
    "DEBUG": ALL_METHODS,
    "pairwise_power": ALL_METHODS,
    "oracle_rates": [],
    "oracle_select_rates": [],
    "bivariate_power": ALL_METHODS,
    "bivariate_multiplic_power": ALL_METHODS,
}


def get_experiment_params(exp):
    return PARAMS_DICT[exp]


def get_param_keys(exp):
    return list(PARAMS_DICT[exp][0].keys())


def get_experiments():
    return list(PARAMS_DICT.keys())


def get_experiment_methods(exp):
    return METHODS_DICT[exp]

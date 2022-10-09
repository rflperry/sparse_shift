import argparse
from pathlib import Path
import logging
import itertools

import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from sparse_shift.datasets import (
    sample_cdnod_sim,
    erdos_renyi_dag,
    connected_erdos_renyi_dag,
    barabasi_albert_dag,
    complete_dag,
)
from sparse_shift.metrics import dag_true_orientations, dag_false_orientations, \
    dag_precision, dag_recall, average_precision_score
from sparse_shift.utils import dag2cpdag, cpdag2dags
from exp_settings import get_experiments, get_experiment_methods, get_experiment_params, get_param_keys

import os
import warnings

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


def _sample_dag(dag_simulator, n_variables, dag_density, seed=None):
    """
    Samples a DAG from a specified distribution
    """
    if dag_simulator == "er":
        dag = erdos_renyi_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "ba":
        dag = barabasi_albert_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == 'complete':
        dag = complete_dag(n_variables)
    else:
        raise ValueError(f"DAG simulator {dag_simulator} not valid optoion")

    count = 0
    if len(cpdag2dags(dag2cpdag(dag))) == 1:
        # Don't sample already solved MECs
        np.random.seed(seed)
        new_seed = int(1000*np.random.uniform())
        dag = _sample_dag(dag_simulator, n_variables, dag_density, new_seed)
        count += 1
        if count > 100:
            raise ValueError(f"Cannot sample a DAG in these settings with nontrivial MEC ({[dag_simulator, n_variables, dag_density]})")

    return dag


def _sample_interventions(n_variables, n_total_environments, sparsity, seed=None):
    np.random.seed(seed)
    if isinstance(sparsity, float):
        sparsity = np.round(n_variables * sparsity).astype(int)
    sampled_targets = [
        np.random.choice(n_variables, sparsity, replace=False)
        for _ in range(n_total_environments)
    ]
    return sampled_targets


def _sample_datasets(data_simulator, sample_size, dag, intervention_targets, seed=None):
    """
    Samples multi-environment data from a specified distribution
    """
    if data_simulator == "cdnod":
        np.random.seed(seed)
        domain_seed = int(1000 * np.random.uniform())
        Xs = [
            sample_cdnod_sim(
                dag,
                sample_size,
                intervention_targets=targets,
                base_random_state=seed,
                domain_random_state=domain_seed + i,
            )
            for i, targets in enumerate(intervention_targets)
        ]
    else:
        raise ValueError(f"Data simulator {data_simulator} not valid optoion")

    return Xs


def main(args):
    # Initialize og details
    logging.basicConfig(
        filename="./logging.log",
        format="%(asctime)s:%(levelname)s:%(message)s",
        level=logging.INFO,
    )
    logging.info(f"NEW RUN: simulation data generation")
    logging.info(f"Args: {args}")
    logging.info(f"Experimental settings:")
    logging.info(get_experiment_params(args.experiment))

    # Construct parameter grids
    param_dicts = get_experiment_params(args.experiment)
    prior_indices = 0
    logging.info(f'{len(param_dicts)} total parameter dictionaries')

    base_path = Path('../../data/')

    for params_dict in param_dicts:
        experiment = params_dict.pop('experiment')
        param_keys, param_values = zip(*params_dict.items())
        params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

        # Iterate over
        write_path = base_path / args.experiment / experiment
        logging.info(f'{experiment}: {len(params_grid)} total parameter combinations')
        logging.info(f'Writing to {write_path}')

        for i, params in enumerate(params_grid):
            logging.info(f"Params {i} / {len(params_grid)}")
            run_experimental_setting(
                args=args,
                params_index=i + prior_indices,
                write_path=write_path,
                **params,
            )
        
        prior_indices += len(params_grid)
    logging.info(f'Complete')


def run_experimental_setting(
    args,
    params_index,
    write_path,
    n_variables,
    n_total_environments,
    sparsity,
    intervention_targets,
    sample_size,
    dag_density,
    reps,
    data_simulator,
    dag_simulator,
):
    if sparsity is not None and sparsity > n_variables:
        logging.info(f"Skipping: sparsity {sparsity} greater than n_variables {n_variables}")
        return

    write_path = write_path / f'params_{params_index}'

    experimental_params = {
        'params_index': params_index,
        'n_variables': n_variables,
        'n_total_environments': n_total_environments,
        'sparsity': sparsity,
        'intervention_targets': intervention_targets,
        'sample_size': sample_size,
        'dag_density': dag_density,
        'reps': reps,
        'data_simulator': data_simulator,
        'dag_simulator': dag_simulator,
    }

    # Save experiment metdata
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    with open(write_path / 'experiment_metadata.dict', 'wb') as f:
        pickle.dump(experimental_params, f)

    def _run_rep(rep):
        results = []
        # Get DAG
        true_dag = _sample_dag(dag_simulator, n_variables, dag_density, seed=rep)
        true_cpdag = dag2cpdag(true_dag)
        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2) // 2

        # Get interventions
        if intervention_targets is None:
            sampled_targets = _sample_interventions(
                n_variables, n_total_environments, sparsity, seed=rep
            )
        else:
            sampled_targets = intervention_targets

        # Sample dataset
        if data_simulator is None:
            return results

        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, sampled_targets, seed=rep
        )

        write_folder = write_path / f'rep_{rep}'

        # Save pvalues
        if not os.path.exists(write_folder):
            os.makedirs(write_folder)

        for env_index, X in enumerate(Xs):
            np.save(write_folder / f"data{env_index+1}.npy", X)

        with open(write_folder / 'ground_truth.dict', 'wb') as f:
            pickle.dump(
                {
                    'true_dag': true_dag,
                    'true_cpdag': true_cpdag,
                    'mec_size': mec_size,
                    'total_edges': total_edges,
                    'unoriented_edges': unoriented_edges,
                },
                f,
            )

    rep_shift = 0
    for rep in tqdm(range(reps)):
        _run_rep(rep + rep_shift)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        help="experiment parameters to run",
        choices=get_experiments(),
    )
    args = parser.parse_args()

    main(args)

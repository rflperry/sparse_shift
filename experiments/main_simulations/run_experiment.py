import argparse
from pathlib import Path
import logging
import pickle
import itertools

import numpy as np
import pandas as pd

from sparse_shift.datasets import (
    sample_cdnod_sim,
    sample_topological,
    erdos_renyi_dag,
    connected_erdos_renyi_dag,
    barabasi_albert_dag,
)
from sparse_shift.plotting import plot_dag
from sparse_shift.testing import test_mechanism_shifts, test_mechanism
from sparse_shift.methods import FullPC, PairwisePC, MinChangeOracle, MinChange
from sparse_shift.metrics import dag_precision, dag_recall
from sparse_shift.utils import dag2cpdag, cpdag2dags
from tqdm import tqdm

import os
import warnings

warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses


EXPERIMENT_DICT = {
    "environment_convergence": {
        "n_variables": [6],
        "n_total_environments": [15],
        "sparsity": [1, 2, 4],
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
        "sample_size": [50, 100, 200, 300, 500],
        "dag_density": [0.3],
        "reps": [20],
        "data_simulator": ["cdnod"],
        "dag_simulator": ["er"],
    }
}


def _sample_dag(dag_simulator, n_variables, dag_density, seed=None):
    """
    Samples a DAG from a specified distribution
    """
    if dag_simulator == "er":
        dag = erdos_renyi_dag(n_variables, dag_density, seed=seed)
    elif dag_simulator == "ba":
        dag = barabasi_albert_dag(n_variables, dag_density, seed=seed)
    else:
        raise ValueError(f"DAG simulator {dag_simulator} not valid optoion")

    if len(cpdag2dags(dag2cpdag(dag))) == 1:
        # Don't sample already solved MECs
        np.random.seed(seed)
        new_seed = int(1000*np.random.uniform())
        dag = _sample_dag(dag_simulator, n_variables, dag_density, new_seed)

    return dag


def _sample_interventions(n_variables, n_total_environments, sparsity, seed=None):
    np.random.seed(seed)
    intervention_targets = [
        np.random.choice(n_variables, sparsity, replace=False)
        for _ in range(n_total_environments)
    ]
    return intervention_targets


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
    logging.info(f"NEW RUN:")
    logging.info(f"Args: {args}")
    logging.info(f"Experimental settings:")
    logging.info(EXPERIMENT_DICT[args.experiment])

    # Construct parameter grids
    param_keys, param_values = zip(*EXPERIMENT_DICT[args.experiment].items())
    params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

    # Create results csv header
    header = np.hstack(
        [
            ["params_index"],
            list(param_keys),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["Number of possible DAGs", "MEC size", "MEC total edges", "MEC unoriented edges"],
            ["Precision", "Recall"],
        ]
    )
    write_file = open(f"./results/{args.experiment}_results.csv", "w+")
    write_file.write(", ".join(header) + "\n")
    write_file.flush()

    # Iterate over
    logging.info(f'{len(params_grid)} total parameter combinations')
    for i, params in enumerate(params_grid):
        logging.info(f"Params ({i} / {len(params_grid)})")
        run_experimental_setting(
            name=args.experiment,
            params_index=i,
            write_file=write_file,
            **params,
        )


def run_experimental_setting(
    name,
    params_index,
    write_file,
    n_variables,
    n_total_environments,
    sparsity,
    sample_size,
    dag_density,
    reps,
    data_simulator,
    dag_simulator,
):

    experimental_params = [
        params_index,
        n_variables,
        n_total_environments,
        sparsity,
        sample_size,
        dag_density,
        reps,
        data_simulator,
        dag_simulator,
    ]
    # experimental_params = [str(val).replace(", ", ";") for val in experimental_params]

    for rep in tqdm(range(reps)):
        # Get DAG
        true_dag = _sample_dag(dag_simulator, n_variables, dag_density, seed=rep)
        true_cpdag = dag2cpdag(true_dag)
        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2)

        # Get interventions
        intervention_targets = _sample_interventions(
            n_variables, n_total_environments, sparsity, seed=rep
        )

        # Compute oracle results
        fpc_oracle = FullPC(true_dag)
        mch_oracle = MinChangeOracle(true_dag)

        for n_env, intv_targets in enumerate(intervention_targets):
            n_env += 1
            fpc_oracle.add_environment(intv_targets)
            mch_oracle.add_environment(intv_targets)

            cpdag = fpc_oracle.get_mec_cpdag()
            write_file.write(
                ", ".join(
                    map(
                        str,
                        experimental_params + [
                            "PC (pool all)",
                            False,
                            n_env,
                            rep,
                            len(fpc_oracle.get_mec_dags()),
                            mec_size,
                            total_edges,
                            unoriented_edges,
                            dag_precision(true_dag, cpdag),
                            dag_recall(true_dag, cpdag),
                        ],
                    )
                )
                + "\n"
            )
            write_file.flush()

            # results_mat.append([
            #     rep, 'PC (pool all)', sparsity, n_env, len(fpc.get_mec_dags()), dag_true_orientations(true_dag, cpdag), dag_false_orientations(true_dag, cpdag)
            # ])

            cpdag = mch_oracle.get_min_cpdag()
            write_file.write(
                ", ".join(
                    map(
                        str,
                        experimental_params + [
                            "Min changes (oracle)",
                            False,
                            n_env,
                            rep,
                            len(mch_oracle.get_min_dags()),
                            mec_size,
                            total_edges,
                            unoriented_edges,
                            dag_precision(true_dag, cpdag),
                            dag_recall(true_dag, cpdag),
                        ],
                    )
                )
                + "\n"
            )
            write_file.flush()
            # results_mat.append([
            #     rep, 'Min changes (oracle)', sparsity, n_env, len(mch_oracle.get_min_dags()), dag_true_orientations(true_dag, cpdag), dag_false_orientations(true_dag, cpdag)
            # ])

        del fpc_oracle, mch_oracle

        # Sample dataset
        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, intervention_targets, seed=rep
        )

        # Compute empirical results
        for save_name, method_name, mch in zip(
            ('mch_kci', 'mch_lin'),
            ('Min changes (kci)', 'Min changes (linear)'),
            (
                MinChange(
                    true_cpdag,
                    alpha=0.05,
                    scale_alpha=True,
                    test='kci',
                    test_kwargs={
                        "KernelX": "GaussianKernel",
                        "KernelY": "GaussianKernel",
                        "KernelZ": "GaussianKernel",
                    },
                ),
                MinChange(
                    true_cpdag, alpha=0.05, scale_alpha=True,
                    test='invariant_residuals',
                    test_kwargs={'method': 'linear', 'test': "whitney_levene"},
                )
            )
        ):

            for n_env, X in enumerate(Xs):
                n_env += 1
                mch.add_environment(X)

                for soft in [True, False]:
                    min_cpdag = mch.get_min_cpdag(soft)
                    write_file.write(
                        ", ".join(
                            map(
                                str,
                                experimental_params + [
                                    method_name,
                                    soft,
                                    n_env,
                                    rep,
                                    len(mch.get_min_dags(soft)),
                                    mec_size,
                                    total_edges,
                                    unoriented_edges,
                                    dag_precision(true_dag, min_cpdag),
                                    dag_recall(true_dag, min_cpdag),
                                ],
                            )
                        )
                        + "\n"
                    )
                    write_file.flush()

                # Save pvalues
                np.save(
                    f"./results/pvalue_mats/{name}_{save_name}_pvalues_params={params_index}_rep={rep}.npy",
                    mch.pvalues_,
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        help="experiment parameters to run",
        choices=list(EXPERIMENT_DICT.keys()),
    )
    args = parser.parse_args()

    main(args)

import argparse
from pathlib import Path
import logging
import pickle
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sparse_shift.datasets import (
    sample_cdnod_sim,
    sample_topological,
    erdos_renyi_dag,
    connected_erdos_renyi_dag,
    barabasi_albert_dag,
    complete_dag,
)
from sparse_shift.plotting import plot_dag
from sparse_shift.testing import test_mechanism_shifts, test_mechanism
from sparse_shift.methods import FullPC, PairwisePC, MinChangeOracle, MinChange
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
    logging.info(f"NEW RUN:")
    logging.info(f"Args: {args}")
    logging.info(f"Experimental settings:")
    logging.info(get_experiment_params(args.experiment))

    # Create results csv header
    header = np.hstack(
        [
            ["params_index"],
            get_param_keys(args.experiment),
            ["Method", "Soft", "Number of environments", "Rep"],
            ["Number of possible DAGs", "MEC size", "MEC total edges", "MEC unoriented edges"],
            ["True orientation rate", "False orientation rate", "Precision", "Recall", 'Average precision'],
        ]
    )
    write_file = open(f"./results/{args.experiment}_results.csv", "w+")
    write_file.write(", ".join(header) + "\n")
    write_file.flush()

    # Construct parameter grids
    param_dicts = get_experiment_params(args.experiment)
    prior_indices = 0
    logging.info(f'{len(param_dicts)} total parameter dictionaries')
    for params_dict in param_dicts:
        param_keys, param_values = zip(*params_dict.items())
        params_grid = [dict(zip(param_keys, v)) for v in itertools.product(*param_values)]

        # Iterate over
        logging.info(f'{len(params_grid)} total parameter combinations')

        for i, params in enumerate(params_grid):
            logging.info(f"Params {i} / {len(params_grid)}")
            run_experimental_setting(
                args=args,
                params_index=i + prior_indices,
                write_file=write_file,
                **params,
            )
        
        prior_indices += len(params_grid)


def run_experimental_setting(
    args,
    params_index,
    write_file,
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

    name = args.experiment

    if sparsity is not None and sparsity > n_variables:
        logging.info(f"Skipping: sparsity {sparsity} greater than n_variables {n_variables}")
        return

    experimental_params = [
        params_index,
        n_variables,
        n_total_environments,
        sparsity,
        intervention_targets,
        sample_size,
        dag_density,
        reps,
        data_simulator,
        dag_simulator,
    ]
    experimental_params = [str(val).replace(", ", ";") for val in experimental_params]

    def _run_rep(rep, write):
        results = []
        # Get DAG
        true_dag = _sample_dag(dag_simulator, n_variables, dag_density, seed=rep)
        true_cpdag = dag2cpdag(true_dag)
        mec_size = len(cpdag2dags(true_cpdag))
        total_edges = np.sum(true_dag)
        unoriented_edges = np.sum((true_cpdag + true_cpdag.T) == 2)

        # Get interventions
        if intervention_targets is None:
            sampled_targets = _sample_interventions(
                n_variables, n_total_environments, sparsity, seed=rep
            )
        else:
            sampled_targets = intervention_targets

        # Compute oracle results
        fpc_oracle = FullPC(true_dag)
        mch_oracle = MinChangeOracle(true_dag)

        for n_env, intv_targets in enumerate(sampled_targets):
            n_env += 1
            fpc_oracle.add_environment(intv_targets)
            mch_oracle.add_environment(intv_targets)

            cpdag = fpc_oracle.get_mec_cpdag()

            true_orients = np.round(dag_true_orientations(true_dag, cpdag), 4)
            false_orients = np.round(dag_false_orientations(true_dag, cpdag), 4)
            precision = np.round(dag_precision(true_dag, cpdag), 4)
            recall = np.round(dag_recall(true_dag, cpdag), 4)
            ap = recall

            result = ", ".join(
                map(
                    str,
                    experimental_params + [
                        "Full PC (oracle)",
                        False,
                        n_env,
                        rep,
                        len(fpc_oracle.get_mec_dags()),
                        mec_size,
                        total_edges,
                        unoriented_edges,
                        true_orients,
                        false_orients,
                        precision,
                        recall,
                        ap,
                    ],
                )
            ) + "\n"
            if write:
                write_file.write(result)
                write_file.flush()
            else:
                results.append(result)

            cpdag = mch_oracle.get_min_cpdag()

            true_orients = np.round(dag_true_orientations(true_dag, cpdag), 4)
            false_orients = np.round(dag_false_orientations(true_dag, cpdag), 4)
            precision = np.round(dag_precision(true_dag, cpdag), 4)
            recall = np.round(dag_recall(true_dag, cpdag), 4)
            ap = recall

            result = ", ".join(
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
                        true_orients,
                        false_orients,
                        precision,
                        recall,
                        ap,
                    ],
                )
            ) + "\n"
            if write:
                write_file.write(result)
                write_file.flush()
            else:
                results.append(result)

        del fpc_oracle, mch_oracle

        # Sample dataset
        if data_simulator is None:
            return results

        Xs = _sample_datasets(
            data_simulator, sample_size, true_dag, sampled_targets, seed=rep
        )

        # Compute empirical results
        for save_name, method_name, mch, hyperparams in get_experiment_methods(
            args.experiment
        ):

            mch = mch(cpdag=true_cpdag, **hyperparams)

            for n_env, X in enumerate(Xs):
                n_env += 1
                mch.add_environment(X)

                for soft in [True, False]:
                    min_cpdag = mch.get_min_cpdag(soft)

                    true_orients = np.round(dag_true_orientations(true_dag, min_cpdag), 4)
                    false_orients = np.round(dag_false_orientations(true_dag, min_cpdag),4 )
                    precision = np.round(dag_precision(true_dag, min_cpdag), 4)
                    recall = np.round(dag_recall(true_dag, min_cpdag), 4)
                    ap = np.round(average_precision_score(true_dag, mch.pvalues_), 4)

                    result = ", ".join(
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
                                true_orients,
                                false_orients,
                                precision,
                                recall,
                                ap,
                            ],
                        )
                    ) + "\n"
                    if write:
                        write_file.write(result)
                        write_file.flush()
                    else:
                        results.append(result)

                # Save pvalues
                np.save(
                    f"./results/pvalue_mats/{name}_{save_name}_pvalues_params={params_index}_rep={rep}.npy",
                    mch.pvalues_,
                )

        return results

    if args.jobs is not None:
        results = Parallel(
                n_jobs=args.jobs,
            )(
                delayed(_run_rep)(rep, False) for rep in range(reps)
            )
        for result in np.concatenate(results):
            write_file.write(result)
        write_file.flush()
    else:
        for rep in tqdm(range(reps)):
            _run_rep(rep, write=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        help="experiment parameters to run",
        choices=get_experiments(),
    )
    parser.add_argument(
        "--jobs",
        help="Number of jobs to run in parallel",
        default=None,
        type=int,
    )
    args = parser.parse_args()

    main(args)

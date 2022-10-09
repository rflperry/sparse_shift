# Overview

Conditional independence tools for causal learning under the sparse mechanism shift hypothesis.

## Local installation

From a clean python environment (e.g. `conda create -n test python=3.9`),

```console
git clone https://github.com/rflperry/sparse_shift.git
cd sparse_shift
pip install -e .
```

## Running experiments and generating figures

First navigate and install necessary packages

```console
cd experiments
pip install -r requirements.txt
```

### Teaser figure

```console
cd experiments

python teaser_sparse_oracle_pc.py
```

Runs the teaser experiment and generates the figure.

### Simulations

```console
cd experiments/main_simulations
```

Then run the following commands to generate results and the camera-ready figures.

Bivariate power:

```console
python run_experiment.py --experiment bivariate_power --quick
python run_experiment.py --experiment bivariate_multiplic_power --quick

python plot_bivariate_identifiability.py
```

Oracle rates:

```console
python run_experiment.py --experiment oracle_rates --quick
python run_experiment.py --experiment oracle_select_rates --quick

python plot_oracle_rates.py

```

Empirical comparison simulations:

```console
python run_experiment.py --experiment pairwise_power --quick

python plot_empirical_power.py

```

Remove `--quick` and add `--n_jobs -2` to run the full paper experiments.
Note that this can take a long time.

### Cytometry experiment

```console
cd experiments/cytometry
python run_cytometry_experiment.py --quick

python analyze_pvalues.py
```

Remove `--quick` and add `--n_jobs -2` to run the full paper experiments.

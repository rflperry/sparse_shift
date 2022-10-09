# Overview

Conditional independence tools for causal learning under the sparse mechanism shift hypothesis.

## Local installation

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
```

Open and run `teaser_sparse_oracle_pc.ipynb` in that folder for experiments and figures.

### Simulations

```console
cd experiments/main_simulations
```

Then run the following commands for bivariate power,

```console
python run_experiment.py --experiment bivariate_power --quick
python run_experiment.py --experiment bivariate_multiplic_power --quick
```

rracle rates,

```console
python run_experiment.py --experiment oracle_rates --quick
python run_experiment.py --experiment oracle_select_rates --quick
```

and empirical comparison simulations.

```console
python run_experiment.py --experiment pairwise_power --quick
```

Remove `--quick` and add `--n_jobs -2` to run the full paper experiments.
Note that this can take a long time.

Then open the three `.ipynb` notebooks to generate the figures.

### Cytometry experiment

```console
cd experiments/cytometry
python run_cytometry_experiment.py --quick
```

Then open and run `analyze_pvalues.ipynb` to generate the figure.

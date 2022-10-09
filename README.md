# Overview

Conditional independence tools for causal learning under the sparse mechanism shift hypothesis.

# Local installation

```console
git clone https://github.com/rflperry/sparse_shift.git
cd sparse_shift
pip install -e .
```

# Package Overview
- data
  - cytometry
    - dataset_NUM.csv [NUM = 1-9]
- experiments
  - cytometry
    - analyze_pvalues.ipynb
    - run_cytometry_experiment.py
    - results/
    - figures/
- notebooks
  - figures/
  - teaser_sparse_oracle_pc.ipynb
- sparse_shift [code]

# Quick replication instructions

## Teaser figure

```console
cd experiments
```

Then open and run `teaser_sparse_oracle_pc.ipynb` in that folder.

## Oracle simulations

```console
cd experiments/main_simulations
```

## MSS comparison simulation

```console
cd experiments/main_simulations
```

## Other method simulations

```console
cd experiments/main_simulations
```



## Cytometry experiment

```console
cd experiments/cytometry
python run_cytometry_experiment.py --quick
```

Then open and run `analyze_pvalues.ipynb` in that folder.

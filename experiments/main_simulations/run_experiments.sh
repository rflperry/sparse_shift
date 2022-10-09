# Bivariate power  (add/multiplicative)
python run_experiment.py --experiment bivariate_power --n_jobs -2
python run_experiment.py --experiment bivariate_multiplic_power --n_jobs -2

# Oracle rates (large select, small everything)
python run_experiment.py --experiment oracle_rates --n_jobs -2
python run_experiment.py --experiment oracle_select_rates --n_jobs -2

# Run method comparisons
python run_experiment.py --experiment pairwise_power --n_jobs -2

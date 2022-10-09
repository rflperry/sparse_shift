# Bivariate power  (add/multiplicative)
python run_experiment.py --experiment bivariate_power --quick
python run_experiment.py --experiment bivariate_multiplic_power --quick

# Oracle rates (large select, small everything)
python run_experiment.py --experiment oracle_rates --quick
python run_experiment.py --experiment oracle_select_rates --quick

# Run method comparisons
python run_experiment.py --experiment pairwise_power --quick

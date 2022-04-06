import numpy as np
from sparse_shift.datasets import sample_nonlinear_icp_sim
import pytest


@pytest.mark.parametrize("nonlinearity", ["id", "relu", "sqrt", "sin"])
@pytest.mark.parametrize("noise_df", [2])
@pytest.mark.parametrize("combination", ["additive", "multiplicative"])
@pytest.mark.parametrize("intervention", ["soft", "hard"])
@pytest.mark.parametrize("intervention_shift", [0, 1])
@pytest.mark.parametrize("intervention_scale", [0, 1])
@pytest.mark.parametrize("intervention_targets", [None, 1, [2, 3]])
@pytest.mark.parametrize("intervention_pct", [None, 0, 0.1])
def test_icp_sim_params_work(
    nonlinearity,
    noise_df,
    combination,
    intervention_targets,
    intervention,
    intervention_shift,
    intervention_scale,
    intervention_pct,
):

    dag = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0]])
    X = sample_nonlinear_icp_sim(
        dag=dag,
        n_samples=100,
        nonlinearity=nonlinearity,
        noise_df=noise_df,
        combination=combination,
        intervention_targets=intervention_targets,
        intervention=intervention,
        intervention_shift=intervention_shift,
        intervention_scale=intervention_scale,
        intervention_pct=intervention_pct,
        random_state=None,
    )

    assert isinstance(X, np.ndarray)
    assert X.shape == (100, dag.shape[0])

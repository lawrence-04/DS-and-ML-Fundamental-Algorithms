import numpy as np

from tp_estimator import TPEstimator


def sphere_function(hyperparameters):
    return np.sum(np.square(hyperparameters))


def test_initialization_and_shapes():
    hp_ranges = [[-5, 5], [-3, 3]]
    tpe = TPEstimator(
        hyperparameter_ranges=hp_ranges, objective_function=sphere_function
    )
    assert tpe.hyperparameters.shape == (25, 2)
    assert tpe.values.shape == (25,)
    assert np.all((tpe.hyperparameters[:, 0] >= -5) & (tpe.hyperparameters[:, 0] <= 5))
    assert np.all((tpe.hyperparameters[:, 1] >= -3) & (tpe.hyperparameters[:, 1] <= 3))


def test_optimisation_returns_valid_result():
    hp_ranges = [[-5, 5], [-3, 3]]
    tpe = TPEstimator(
        hyperparameter_ranges=hp_ranges, objective_function=sphere_function, num_iter=5
    )
    best = tpe.optimise()
    assert best.shape == (2,)
    assert np.all((best[0] >= -5) & (best[0] <= 5))
    assert np.all((best[1] >= -3) & (best[1] <= 3))


def test_improvement_after_optimisation():
    hp_ranges = [[-5, 5], [-5, 5]]
    tpe = TPEstimator(
        hyperparameter_ranges=hp_ranges, objective_function=sphere_function, num_iter=10
    )
    initial_best = np.min(tpe.values)
    tpe.optimise()
    final_best = np.min(tpe.values)
    assert final_best <= initial_best

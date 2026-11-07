import numpy as np
import pandas as pd
from backend.data_loader import calculate_s11_db
from backend.fitness_function import make_fitness
from backend.differential_evolution import differential_evolution
from backend.gwo import gwo
from backend.pso import pso


def test_s11_zero_is_finite():
    result = calculate_s11_db([0, 1], [0, 0])
    assert np.isfinite(result).all()
    assert result[1] == 0


def test_optimizers_improve_quadratic():
    bounds = ([-5, -5], [5, 5])
    objective = lambda x: float(np.sum(np.asarray(x) ** 2))
    for algorithm in (pso, differential_evolution, gwo):
        result = algorithm(objective, bounds, population_size=12, iterations=15, seed=3)
        assert result["best_fitness"] < 3
        assert len(result["history"]) >= 2


class _CurveModel:
    features = ["x"]
    targets = ["S11_dB_0", "S11_dB_1", "S11_dB_2"]

    def predict(self, values):
        x = float(values[0][0])
        return np.array([[-20.0 + x, -5.0 + x, -12.0 + x]])


def test_fitness_uses_complete_s11_curve():
    fitness = make_fitness(_CurveModel(), ["x"], ([-1], [1]))
    assert fitness([0]) < fitness([1])
    assert fitness([1]) > fitness([0])


def test_fitness_rejects_mismatched_frequency_metadata():
    try:
        make_fitness(
            _CurveModel(), ["x"], ([-1], [1]),
            frequency_values=[1.0, 2.0],
        )([0])
    except ValueError as exc:
        assert "frequency" in str(exc).lower()
    else:
        raise AssertionError("Expected mismatched frequency metadata to fail.")

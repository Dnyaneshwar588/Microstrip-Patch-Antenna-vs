import numpy as np
import pandas as pd
from backend.data_loader import calculate_s11_db
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

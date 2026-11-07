import numpy as np
import pandas as pd
from backend.data_loader import calculate_s11_db
from backend.fitness_function import make_fitness
from backend.frequency_metadata import load_frequency_values
from backend.performance_metrics import response_metrics
from backend.result_comparison import compare_s11_results
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


def test_frequency_range_is_explicit_and_converted_to_hz():
    values = load_frequency_values(start=2, stop=3, count=3, unit="GHz")
    np.testing.assert_allclose(values, [2e9, 2.5e9, 3e9])


def test_response_metrics_requires_frequency_for_resonance():
    metrics = response_metrics(np.array([-3.0, -12.0, -8.0]))
    assert metrics["minimum_s11_db"] == -12.0
    assert metrics["frequency_available"] is False
    with_frequency = response_metrics(
        np.array([-3.0, -12.0, -8.0]),
        np.array([2e9, 2.5e9, 3e9]),
    )
    assert with_frequency["resonant_frequency_hz"] == 2.5e9
    assert with_frequency["bandwidth_hz"] == 0.0


def test_hfss_comparison_interpolates_actual_export():
    comparison = compare_s11_results(
        np.array([2e9, 2.5e9, 3e9]),
        np.array([-3.0, -12.0, -8.0]),
        pd.DataFrame({"Frequency (GHz)": [2.0, 3.0], "S11 (dB)": [-2.0, -7.0]}),
    )
    assert comparison["mae_db"] > 0
    assert len(comparison["frequency_hz"]) == 3

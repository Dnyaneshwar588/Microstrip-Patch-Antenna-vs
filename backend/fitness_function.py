from __future__ import annotations
import numpy as np


def make_fitness(model, features, bounds, target_frequency=None, frequency_values=None, weights=None):
    weights = {
        "s11": 1.0, "frequency": 1.0, "vswr": 1.0, "gain": 1.0, "bandwidth": 1.0,
        **(weights or {}),
    }
    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)
    if lower.shape != upper.shape or lower.shape[0] != len(features):
        raise ValueError("Optimization bounds must match the model features.")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)) or np.any(lower > upper):
        raise ValueError("Optimization bounds must be finite and ordered.")
    frequencies = None if frequency_values is None else np.asarray(frequency_values, dtype=float)

    def fitness(vector):
        vector = np.asarray(vector, dtype=float)
        if np.any(vector < lower) or np.any(vector > upper):
            return 1e9
        prediction = np.asarray(model.predict([vector])).ravel()
        names = {name.lower(): value for name, value in zip(model.targets, prediction)}
        s11_names = [key for key in model.targets if "s11" in key.lower()]
        s11_values = np.asarray([names[key.lower()] for key in s11_names], dtype=float)
        if not len(s11_values) or np.any(~np.isfinite(s11_values)):
            return 1e9
        if frequencies is not None and len(frequencies) != len(s11_values):
            raise ValueError("Frequency metadata must match the predicted S11 response length.")
        if target_frequency is not None and frequencies is not None:
            target_index = int(np.argmin(np.abs(frequencies - target_frequency)))
            target_s11 = s11_values[target_index]
            frequency_error = abs(float(frequencies[target_index]) - target_frequency)
        else:
            target_s11 = float(np.min(s11_values))
            frequency_error = 0.0
        magnitude = np.power(10.0, s11_values / 20.0)
        vswr = (1.0 + magnitude) / np.maximum(1.0 - magnitude, np.finfo(float).eps)
        score = weights["s11"] * max(0.0, target_s11 + 10.0)
        score += weights["vswr"] * max(0.0, float(np.max(vswr)) - 2.0)
        score += weights["frequency"] * frequency_error
        if frequencies is not None and len(frequencies) > 1:
            score -= weights["bandwidth"] * _bandwidth(frequencies, s11_values, -10.0)
        gain = next((v for k, v in names.items() if "gain" in k), None)
        if gain is not None:
            score -= weights["gain"] * gain
        return float(score)
    return fitness


def _bandwidth(frequencies: np.ndarray, s11_db: np.ndarray, threshold: float) -> float:
    valid = s11_db <= threshold
    if not np.any(valid):
        return 0.0
    return float(frequencies[valid].max() - frequencies[valid].min())

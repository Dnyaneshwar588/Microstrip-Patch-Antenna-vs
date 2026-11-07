from __future__ import annotations
import numpy as np


def make_fitness(model, features, bounds, target_frequency=None, weights=None):
    weights = weights or {"s11": 1.0, "frequency": 1.0, "vswr": 1.0, "gain": 1.0, "bandwidth": 1.0}
    lower, upper = np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float)

    def fitness(vector):
        vector = np.asarray(vector, dtype=float)
        if np.any(vector < lower) or np.any(vector > upper):
            return 1e9
        prediction = np.asarray(model.predict([vector])).ravel()
        names = {name.lower(): value for name, value in zip(model.targets, prediction)}
        score = 0.0
        s11 = next((v for k, v in names.items() if "s11" in k), None)
        if s11 is not None:
            score += weights["s11"] * max(0.0, s11 + 10.0)
        frequency = next((v for k, v in names.items() if "freq" in k), None)
        if target_frequency is not None and frequency is not None:
            score += weights["frequency"] * abs(frequency - target_frequency)
        vswr = next((v for k, v in names.items() if "vswr" in k), None)
        if vswr is not None:
            score += weights["vswr"] * max(0.0, vswr - 2.0)
        gain = next((v for k, v in names.items() if "gain" in k), None)
        if gain is not None:
            score -= weights["gain"] * gain
        return float(score)
    return fitness

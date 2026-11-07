from __future__ import annotations

import numpy as np


def response_metrics(
    s11_db: np.ndarray,
    frequencies_hz: np.ndarray | None = None,
    threshold_db: float = -10.0,
) -> dict[str, object]:
    values = np.asarray(s11_db, dtype=float).reshape(-1)
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("S11 response must contain finite values.")
    magnitude = np.power(10.0, values / 20.0)
    vswr = (1.0 + magnitude) / np.maximum(1.0 - magnitude, np.finfo(float).eps)
    result: dict[str, object] = {
        "minimum_s11_db": float(np.min(values)),
        "minimum_s11_index": int(np.argmin(values)),
        "reflection_coefficient_magnitude": magnitude.tolist(),
        "vswr": vswr.tolist(),
        "frequency_available": frequencies_hz is not None,
    }
    if frequencies_hz is None:
        result["note"] = "Frequency metadata is unavailable; resonance and bandwidth are disabled."
        return result
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if len(frequencies) != len(values):
        raise ValueError("Frequency and S11 arrays must have the same length.")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("Frequency values must be strictly increasing.")
    passing = values <= threshold_db
    result.update({
        "resonant_frequency_hz": float(frequencies[np.argmin(values)]),
        "bandwidth_hz": float(frequencies[passing].max() - frequencies[passing].min())
        if np.any(passing) else 0.0,
        "fractional_bandwidth": float(
            (frequencies[passing].max() - frequencies[passing].min())
            / frequencies[np.argmin(values)]
        ) if np.any(passing) else 0.0,
        "frequency_hz": frequencies.tolist(),
        "threshold_db": threshold_db,
    })
    return result

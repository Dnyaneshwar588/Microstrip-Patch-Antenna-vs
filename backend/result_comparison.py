from __future__ import annotations

import numpy as np
import pandas as pd

from .performance_metrics import response_metrics


def compare_s11_results(
    predicted_frequency_hz: np.ndarray,
    predicted_s11_db: np.ndarray,
    hfss_frame: pd.DataFrame,
) -> dict[str, object]:
    """Compare a predicted response with actual HFSS columns on matching frequencies."""
    columns = {str(column).strip().lower(): column for column in hfss_frame.columns}
    frequency_column = next(
        (column for key, column in columns.items() if "freq" in key), None
    )
    s11_column = next(
        (column for key, column in columns.items() if "s11" in key and "db" in key), None
    )
    if frequency_column is None or s11_column is None:
        raise ValueError("HFSS results must contain frequency and S11 dB columns.")
    frequency = pd.to_numeric(hfss_frame[frequency_column], errors="coerce").to_numpy()
    actual = pd.to_numeric(hfss_frame[s11_column], errors="coerce").to_numpy()
    valid = np.isfinite(frequency) & np.isfinite(actual)
    if valid.sum() < 2:
        raise ValueError("HFSS results contain fewer than two valid frequency points.")
    frequency, actual = frequency[valid], actual[valid]
    predicted_frequency_hz = np.asarray(predicted_frequency_hz, dtype=float)
    predicted_s11_db = np.asarray(predicted_s11_db, dtype=float)
    if np.nanmax(frequency) < 1e6:
        frequency = frequency * 1e9
    order = np.argsort(frequency)
    frequency, actual = frequency[order], actual[order]
    inside = (predicted_frequency_hz >= frequency.min()) & (predicted_frequency_hz <= frequency.max())
    if not np.any(inside):
        raise ValueError("Predicted and HFSS frequency ranges do not overlap.")
    interpolated = np.interp(predicted_frequency_hz[inside], frequency, actual)
    predicted = predicted_s11_db[inside]
    error = predicted - interpolated
    return {
        "frequency_hz": predicted_frequency_hz[inside].tolist(),
        "predicted_s11_db": predicted.tolist(),
        "hfss_s11_db": interpolated.tolist(),
        "mae_db": float(np.mean(np.abs(error))),
        "rmse_db": float(np.sqrt(np.mean(error ** 2))),
        "predicted_metrics": response_metrics(predicted, predicted_frequency_hz[inside]),
        "hfss_metrics": response_metrics(interpolated, predicted_frequency_hz[inside]),
    }

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


UNIT_SCALE = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def load_frequency_values(
    source: str | Path | BinaryIO | None = None,
    *,
    start: float | None = None,
    stop: float | None = None,
    count: int | None = None,
    unit: str = "GHz",
) -> np.ndarray | None:
    """Load an explicit frequency vector or create one from a verified sweep range."""
    normalized_unit = unit.strip().lower()
    if normalized_unit not in UNIT_SCALE:
        raise ValueError("Frequency unit must be Hz, kHz, MHz, or GHz.")
    if source is not None and (start is not None or stop is not None):
        raise ValueError("Provide either a frequency file or a configured range, not both.")
    if source is not None:
        name = str(getattr(source, "name", source)).lower()
        frame = pd.read_excel(source) if name.endswith((".xlsx", ".xls")) else pd.read_csv(source)
        numeric = frame.select_dtypes(include="number")
        if numeric.shape[1] != 1:
            raise ValueError("Frequency metadata must contain exactly one numeric column.")
        values = numeric.iloc[:, 0].to_numpy(dtype=float)
    elif start is not None or stop is not None:
        if start is None or stop is None or count is None:
            raise ValueError("Frequency range requires start, stop, and point count.")
        if count < 2 or not float(start) < float(stop):
            raise ValueError("Frequency range must have at least two increasing points.")
        values = np.linspace(float(start), float(stop), int(count))
    else:
        return None
    values = values * UNIT_SCALE[normalized_unit]
    if len(values) < 2 or np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0):
        raise ValueError("Frequency values must be finite and strictly increasing.")
    return values


def validate_frequency_values(values: np.ndarray | None, expected_count: int) -> None:
    if values is None:
        return
    if len(values) != expected_count:
        raise ValueError(
            f"Frequency metadata has {len(values)} points but the S11 response has "
            f"{expected_count} points."
        )

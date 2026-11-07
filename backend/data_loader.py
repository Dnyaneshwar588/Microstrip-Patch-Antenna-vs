from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
import numpy as np
import pandas as pd
from .frequency_metadata import load_frequency_values, validate_frequency_values


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    input_columns: list[str]
    target_columns: list[str]
    frequency_columns: list[str]
    warnings: list[str]
    metadata: dict[str, object]
    frequency_values_hz: np.ndarray | None = None


def _read(source: str | Path | BinaryIO) -> pd.DataFrame:
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    return pd.read_csv(source)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def _frequency_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if any(token in str(c).lower() for token in ("freq", "ghz", "mhz"))]


def load_dataset(
    inputs: str | Path | BinaryIO,
    s11_real: str | Path | BinaryIO | None = None,
    s11_imag: str | Path | BinaryIO | None = None,
    frequency_source: str | Path | BinaryIO | None = None,
    frequency_start: float | None = None,
    frequency_stop: float | None = None,
    frequency_count: int | None = None,
    frequency_unit: str = "GHz",
) -> DatasetBundle:
    """Load a single table or align input/complex S11 files by row index."""
    warnings: list[str] = []
    input_frame = _read(inputs).reset_index(drop=True)
    if input_frame.empty:
        raise ValueError("The input dataset is empty.")
    if input_frame.columns.duplicated().any():
        raise ValueError("The input dataset contains duplicate column names.")
    if s11_real is None and s11_imag is None:
        numeric = _numeric_columns(input_frame)
        numeric_frame = input_frame[numeric]
        metadata = _metadata(input_frame, numeric_frame, warnings)
        targets = [column for column in numeric if "s11" in str(column).lower()]
        features = [column for column in numeric if column not in targets]
        metadata["s11_points"] = len(targets)
        frequencies = load_frequency_values(
            frequency_source, start=frequency_start, stop=frequency_stop,
            count=frequency_count, unit=frequency_unit,
        )
        validate_frequency_values(frequencies, len(targets))
        return DatasetBundle(
            input_frame,
            features or numeric,
            targets,
            _frequency_columns(targets),
            warnings,
            metadata,
            frequencies,
        )
    if s11_real is None or s11_imag is None:
        raise ValueError("Both S11 real and S11 imaginary files are required.")
    real = _read(s11_real).reset_index(drop=True)
    imag = _read(s11_imag).reset_index(drop=True)
    if real.empty or imag.empty:
        raise ValueError("S11 component files must not be empty.")
    if len(real) != len(imag) or len(input_frame) != len(real):
        raise ValueError(
            f"Cannot align files by antenna sample: inputs={len(input_frame)}, "
            f"real={len(real)}, imaginary={len(imag)}."
        )
    real_num, imag_num = _numeric_columns(real), _numeric_columns(imag)
    if len(real_num) != len(imag_num) or set(map(str, real_num)) != set(map(str, imag_num)):
        warnings.append("S11 component columns differ; aligned numeric columns by position.")
    count = min(len(real_num), len(imag_num))
    output_columns: dict[str, pd.Series] = {}
    targets: list[str] = []
    for index in range(count):
        label = str(real_num[index])
        target = f"S11_dB_{label}"
        magnitude = np.hypot(pd.to_numeric(real[real_num[index]], errors="coerce"),
                             pd.to_numeric(imag[imag_num[index]], errors="coerce"))
        output_columns[target] = 20 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))
        targets.append(target)
    if count == 0:
        raise ValueError("S11 files contain no numeric columns.")
    if not _frequency_columns(real_num):
        warnings.append(
            f"The S11 files contain {count} sweep columns but no frequency labels. "
            "S11 dB values are valid by sweep index; resonant frequency and bandwidth "
            "cannot be derived until frequency metadata is provided."
        )
    result = pd.concat([input_frame, pd.DataFrame(output_columns)], axis=1)
    frequencies = load_frequency_values(
        frequency_source, start=frequency_start, stop=frequency_stop,
        count=frequency_count, unit=frequency_unit,
    )
    validate_frequency_values(frequencies, count)
    metadata = _metadata(result, _numeric_columns(result), warnings)
    metadata.update({
        "input_rows": len(input_frame),
        "s11_points": count,
        "frequency_labels_verified": bool(_frequency_columns(real_num)),
    })
    metadata["frequency_available"] = frequencies is not None
    return DatasetBundle(
        result, _numeric_columns(input_frame), targets, _frequency_columns(real_num),
        warnings, metadata, frequencies,
    )


def _metadata(frame: pd.DataFrame, numeric_columns: list[str], warnings: list[str]) -> dict[str, object]:
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    missing = int(frame.isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    if missing:
        warnings.append(f"Dataset contains {missing} missing values.")
    return {
        "rows": len(frame),
        "columns": len(frame.columns),
        "numeric_columns": len(numeric_columns),
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "non_numeric_columns": [str(c) for c in frame.columns if c not in numeric_columns],
        "numeric_ranges": {
            str(column): {"min": float(numeric[column].min()), "max": float(numeric[column].max())}
            for column in numeric_columns
            if numeric[column].notna().any()
        },
    }


def calculate_s11_db(real: pd.Series | np.ndarray, imag: pd.Series | np.ndarray) -> np.ndarray:
    magnitude = np.hypot(np.asarray(real, dtype=float), np.asarray(imag, dtype=float))
    return 20 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))

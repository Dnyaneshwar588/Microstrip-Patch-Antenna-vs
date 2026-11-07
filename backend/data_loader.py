from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    input_columns: list[str]
    target_columns: list[str]
    frequency_columns: list[str]
    warnings: list[str]


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
) -> DatasetBundle:
    """Load a single table or align input/complex S11 files by row index."""
    warnings: list[str] = []
    input_frame = _read(inputs).reset_index(drop=True)
    if s11_real is None and s11_imag is None:
        numeric = _numeric_columns(input_frame)
        return DatasetBundle(input_frame, numeric, [], _frequency_columns(numeric), warnings)
    if s11_real is None or s11_imag is None:
        raise ValueError("Both S11 real and S11 imaginary files are required.")
    real = _read(s11_real).reset_index(drop=True)
    imag = _read(s11_imag).reset_index(drop=True)
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
    return DatasetBundle(result, _numeric_columns(input_frame), targets, _frequency_columns(real_num), warnings)


def calculate_s11_db(real: pd.Series | np.ndarray, imag: pd.Series | np.ndarray) -> np.ndarray:
    magnitude = np.hypot(np.asarray(real, dtype=float), np.asarray(imag, dtype=float))
    return 20 * np.log10(np.maximum(magnitude, np.finfo(float).tiny))

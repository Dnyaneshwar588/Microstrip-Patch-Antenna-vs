from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    transformer: ColumnTransformer
    report: dict[str, object]


def prepare_data(
    frame: pd.DataFrame,
    features: list[str],
    targets: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    if not features or not targets:
        raise ValueError("Select at least one numeric feature and target.")
    missing = set(features + targets) - set(frame.columns)
    if missing:
        raise ValueError(f"Unknown columns: {', '.join(sorted(missing))}")
    selected = frame[features + targets].copy()
    if len(set(features)) != len(features) or len(set(targets)) != len(targets):
        raise ValueError("Features and targets must not contain duplicate columns.")
    if set(features) & set(targets):
        raise ValueError("A column cannot be selected as both a feature and a target.")
    selected = selected.apply(pd.to_numeric, errors="coerce")
    valid = selected.dropna(subset=targets)
    if len(valid) < 5:
        raise ValueError("At least five rows with valid target values are required.")
    X, y = valid[features], valid[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    transformer = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scaler", StandardScaler())]), features)],
        remainder="drop",
    )
    transformer.fit(X_train)
    report = {
        "rows": len(selected),
        "usable_rows": len(valid),
        "missing_values": int(selected.isna().sum().sum()),
        "duplicate_rows": int(selected.duplicated().sum()),
        "outliers": int(((X - X.median()).abs() > 3 * X.std()).sum().sum()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    return PreparedData(X_train, X_test, y_train, y_test, transformer, report)

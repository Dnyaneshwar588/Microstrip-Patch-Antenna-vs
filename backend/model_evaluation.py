from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(models, prepared) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        prediction = model.model.predict(prepared.X_test)
        rows.append({
            "model": name,
            "r2": r2_score(prepared.y_test, prediction, multioutput="uniform_average"),
            "rmse": float(np.sqrt(mean_squared_error(prepared.y_test, prediction))),
            "mae": mean_absolute_error(prepared.y_test, prediction),
            "explained_variance": explained_variance_score(
                prepared.y_test, prediction, multioutput="uniform_average"
            ),
        })
    return pd.DataFrame(rows).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)

from __future__ import annotations

from dataclasses import dataclass
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


@dataclass
class TrainedModel:
    name: str
    model: Pipeline
    features: list[str]
    targets: list[str]

    def predict(self, values):
        if not isinstance(values, pd.DataFrame):
            values = pd.DataFrame(values, columns=self.features)
        result = np.asarray(self.model.predict(values))
        return result.reshape(result.shape[0], -1)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class ReducedOutputRegressor(BaseEstimator, RegressorMixin):
    """Fit a regressor to a compact representation of a wide response curve."""

    def __init__(self, estimator, n_components=32):
        self.estimator = estimator
        self.n_components = n_components

    def fit(self, X, y):
        self.output_count_ = y.shape[1] if hasattr(y, "shape") and len(y.shape) > 1 else 1
        self.pca_ = PCA(n_components=min(self.n_components, self.output_count_), random_state=42)
        transformed = self.pca_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, transformed)
        return self

    def predict(self, X):
        return self.pca_.inverse_transform(self.estimator_.predict(X))


def make_estimators(random_state: int = 42) -> dict[str, object]:
    return {
        "Random Forest": RandomForestRegressor(n_estimators=60, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=60, max_depth=2, random_state=random_state)
        ),
        "SVR": MultiOutputRegressor(SVR()),
        "ANN": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=300, early_stopping=True,
                            random_state=random_state),
    }


def train_models(prepared, random_state: int = 42) -> dict[str, TrainedModel]:
    models = {}
    wide_output = prepared.y_train.shape[1] > 32
    for name, estimator in make_estimators(random_state).items():
        if wide_output:
            estimator = ReducedOutputRegressor(estimator, n_components=12)
        pipeline = Pipeline([("features", prepared.transformer), ("model", estimator)])
        target = prepared.y_train.iloc[:, 0] if prepared.y_train.shape[1] == 1 and name in {"Random Forest", "ANN"} else prepared.y_train
        pipeline.fit(prepared.X_train, target)
        models[name] = TrainedModel(name, pipeline, list(prepared.X_train.columns),
                                    list(prepared.y_train.columns))
    return models

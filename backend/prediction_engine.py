from __future__ import annotations
import numpy as np


def predict(model, values):
    prediction = np.asarray(model.predict([values])).ravel()
    return dict(zip(model.targets, prediction))


def range_warnings(values, training_frame):
    warnings = []
    for column, value in values.items():
        if column in training_frame:
            low, high = training_frame[column].min(), training_frame[column].max()
            if value < low or value > high:
                warnings.append(f"{column}={value:g} is outside the training range [{low:g}, {high:g}].")
    return warnings

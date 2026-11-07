from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.data_loader import load_dataset
from backend.performance_metrics import response_metrics
from backend.prediction_engine import prediction_csv
from backend.result_comparison import compare_s11_results
from backend.data_preprocessing import prepare_data
from backend.fitness_function import make_fitness
from backend.hfss_integration import load_manual_results
from backend.ml_models import train_models
from backend.model_evaluation import evaluate_models
from backend.optimization_engine import run_algorithm
from backend.prediction_engine import predict, range_warnings
from backend.report_generator import make_pdf

app = FastAPI(title="Antenna Optimization API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict[str, object] = {
    "bundle": None,
    "prepared": None,
    "models": {},
    "evaluation": None,
    "optimizations": {},
    "hfss_results": None,
}


def _named_buffer(content: bytes, filename: str) -> io.BytesIO:
    buffer = io.BytesIO(content)
    buffer.name = filename
    return buffer


def _load_bundled_dataset() -> None:
    root = Path(__file__).parent / "Microstrip-Patch-main"
    inputs = root / "input_parameters.xlsx"
    real = root / "S11_re.xlsx"
    imag = root / "S11_im.xlsx"
    if not all(path.exists() for path in (inputs, real, imag)):
        return
    bundle = load_dataset(inputs, real, imag)
    state["bundle"] = bundle


def _frame_payload(frame: pd.DataFrame, limit: int = 100) -> dict[str, object]:
    preview = frame.head(limit).where(pd.notna(frame.head(limit)), None).to_dict(orient="records")
    return {"columns": list(frame.columns), "rows": preview, "total": len(frame)}


def _require_bundle():
    if state["bundle"] is None:
        raise HTTPException(400, "Load a dataset before using this feature.")
    return state["bundle"]


def _require_models():
    if not state["models"] or state["prepared"] is None:
        raise HTTPException(400, "Train at least one model before using this feature.")


class TrainRequest(BaseModel):
    features: list[str] = Field(min_length=1)
    targets: list[str] = Field(min_length=1)


class PredictRequest(BaseModel):
    model: str
    values: dict[str, float]


class OptimizeRequest(BaseModel):
    model: str
    algorithm: str
    population: int = Field(default=25, ge=5, le=100)
    iterations: int = Field(default=30, ge=1, le=200)
    bounds: dict[str, list[float]]
    target_frequency_hz: float | None = Field(default=None, gt=0)
    desired_min_s11_db: float = Field(default=-10.0, le=0)
    minimum_bandwidth_hz: float = Field(default=0.0, ge=0)
    maximum_vswr: float = Field(default=2.0, gt=1)
    weights: dict[str, float] = Field(default_factory=dict)


class CompareRequest(OptimizeRequest):
    trials: int = Field(default=3, ge=1, le=20)
    seed: int = Field(default=42, ge=0)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/state")
def get_state():
    if state["bundle"] is None:
        _load_bundled_dataset()
    bundle = state["bundle"]
    evaluation = state["evaluation"]
    return {
        "dataset": None if bundle is None else {
            "rows": len(bundle.data),
            "columns": list(bundle.data.columns),
            "inputColumns": bundle.input_columns,
            "targetColumns": bundle.target_columns,
            "warnings": bundle.warnings,
            "metadata": bundle.metadata,
            "frequencyHz": None if bundle.frequency_values_hz is None else bundle.frequency_values_hz.tolist(),
            "preview": _frame_payload(bundle.data),
        },
        "models": list(state["models"]),
        "evaluation": None if evaluation is None else evaluation.to_dict(orient="records"),
        "optimizations": state["optimizations"],
    }


@app.post("/api/dataset")
async def upload_dataset(
    inputs: Annotated[UploadFile, File(...)],
    s11_real: Annotated[UploadFile | None, File()] = None,
    s11_imag: Annotated[UploadFile | None, File()] = None,
    frequency: Annotated[UploadFile | None, File()] = None,
    frequency_start: Annotated[float | None, Form()] = None,
    frequency_stop: Annotated[float | None, Form()] = None,
    frequency_count: Annotated[int | None, Form()] = None,
    frequency_unit: Annotated[str, Form()] = "GHz",
):
    try:
        bundle = load_dataset(
            _named_buffer(await inputs.read(), inputs.filename or "inputs.xlsx"),
            _named_buffer(await s11_real.read(), s11_real.filename or "s11_real.xlsx") if s11_real else None,
            _named_buffer(await s11_imag.read(), s11_imag.filename or "s11_imag.xlsx") if s11_imag else None,
            frequency_source=_named_buffer(await frequency.read(), frequency.filename or "frequency.csv") if frequency else None,
            frequency_start=frequency_start,
            frequency_stop=frequency_stop,
            frequency_count=frequency_count,
            frequency_unit=frequency_unit,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    state.update(bundle=bundle, prepared=None, models={}, evaluation=None, optimizations={})
    return get_state()


@app.post("/api/train")
def train(request: TrainRequest):
    bundle = _require_bundle()
    try:
        prepared = prepare_data(bundle.data, request.features, request.targets)
        models = train_models(prepared)
        evaluation = evaluate_models(models, prepared)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state.update(prepared=prepared, models=models, evaluation=evaluation, optimizations={})
    return get_state()


@app.post("/api/predict")
def make_prediction(request: PredictRequest):
    _require_models()
    models = state["models"]
    if request.model not in models:
        raise HTTPException(404, "Unknown model.")
    model = models[request.model]
    missing = set(model.features) - set(request.values)
    if missing:
        raise HTTPException(400, f"Missing features: {', '.join(sorted(missing))}")
    values = {feature: request.values[feature] for feature in model.features}
    if any(not math.isfinite(value) for value in values.values()):
        raise HTTPException(400, "Prediction values must be finite numbers.")
    prediction = predict(model, values)
    return {
        "warnings": range_warnings(values, state["prepared"].X_train),
        "prediction": prediction,
        "metrics": response_metrics(
            list(prediction.values()),
            state["bundle"].frequency_values_hz,
        ),
    }


@app.post("/api/predict.csv")
def download_prediction(request: PredictRequest):
    _require_models()
    models = state["models"]
    if request.model not in models:
        raise HTTPException(404, "Unknown model.")
    model = models[request.model]
    missing = set(model.features) - set(request.values)
    if missing:
        raise HTTPException(400, f"Missing features: {', '.join(sorted(missing))}")
    values = {feature: request.values[feature] for feature in model.features}
    return Response(
        prediction_csv(model, values, state["bundle"].frequency_values_hz),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicted_s11.csv"},
    )


@app.post("/api/optimize")
def optimize(request: OptimizeRequest):
    _require_models()
    models = state["models"]
    if request.model not in models:
        raise HTTPException(404, "Unknown model.")
    model = models[request.model]
    if set(request.bounds) != set(model.features):
        raise HTTPException(400, "Bounds must be supplied for every model feature.")
    if any(len(request.bounds[feature]) != 2 for feature in model.features):
        raise HTTPException(400, "Each optimization bound must contain exactly [lower, upper].")
    lower = [request.bounds[feature][0] for feature in model.features]
    upper = [request.bounds[feature][1] for feature in model.features]
    if any(
        not math.isfinite(value)
        for value in [*lower, *upper]
    ):
        raise HTTPException(400, "Optimization bounds must be finite numbers.")
    if any(low > high for low, high in zip(lower, upper)):
        raise HTTPException(400, "Optimization lower bounds must not exceed upper bounds.")
    try:
        bundle = state["bundle"]
        if request.target_frequency_hz is not None and bundle.frequency_values_hz is None:
            raise HTTPException(400, "Target frequency requires verified frequency metadata.")
        weights = {
            "s11": 1.0, "frequency": 1.0, "vswr": 1.0, "bandwidth": 1.0,
            **request.weights,
        }
        objective = make_fitness(
            model, model.features, (lower, upper),
            target_frequency=request.target_frequency_hz,
            frequency_values=bundle.frequency_values_hz,
            weights=weights,
        )
        result = run_algorithm(
            request.algorithm, objective, (lower, upper),
            request.population, request.iterations, 42,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    serializable = {
        **result,
        "best_position": result["best_position"].tolist(),
    }
    state["optimizations"][request.algorithm] = serializable
    return serializable


@app.post("/api/compare")
def compare(request: CompareRequest):
    _require_models()
    if request.model not in state["models"]:
        raise HTTPException(404, "Unknown model.")
    if request.algorithm not in {"PSO", "DE", "GWO"}:
        raise HTTPException(400, "Algorithm must be PSO, DE, or GWO.")
    model = state["models"][request.model]
    if set(request.bounds) != set(model.features):
        raise HTTPException(400, "Bounds must be supplied for every model feature.")
    lower = [request.bounds[feature][0] for feature in model.features]
    upper = [request.bounds[feature][1] for feature in model.features]
    bundle = state["bundle"]
    if request.target_frequency_hz is not None and bundle.frequency_values_hz is None:
        raise HTTPException(400, "Target frequency requires verified frequency metadata.")
    objective = make_fitness(
        model, model.features, (lower, upper),
        target_frequency=request.target_frequency_hz,
        frequency_values=bundle.frequency_values_hz,
        weights=request.weights,
    )
    rows = []
    for algorithm in ("PSO", "DE", "GWO"):
        trials = [
            run_algorithm(algorithm, objective, (lower, upper),
                          request.population, request.iterations, request.seed + index)
            for index in range(request.trials)
        ]
        scores = [float(item["best_fitness"]) for item in trials]
        rows.append({
            "algorithm": algorithm,
            "best_fitness": min(scores),
            "average_fitness": float(pd.Series(scores).mean()),
            "worst_fitness": max(scores),
            "std_fitness": float(pd.Series(scores).std(ddof=0)),
            "elapsed_seconds": float(sum(item["elapsed_seconds"] for item in trials)),
            "objective_evaluations": sum(item["objective_evaluations"] for item in trials),
            "history": trials[0]["history"],
            "best_position": trials[scores.index(min(scores))]["best_position"].tolist(),
        })
    return {"results": rows, "best_algorithm": min(rows, key=lambda row: row["average_fitness"])["algorithm"]}


@app.post("/api/hfss")
async def upload_hfss(result: Annotated[UploadFile, File(...)]):
    try:
        frame = load_manual_results(_named_buffer(await result.read(), result.filename or "hfss.csv"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state["hfss_results"] = frame
    return {**_frame_payload(frame), "validation_status": "Imported actual HFSS export; comparison pending."}


@app.post("/api/hfss/compare")
async def compare_hfss(
    result: Annotated[UploadFile, File(...)],
    model_name: Annotated[str, Form(...)],
    values_json: Annotated[str, Form(...)],
):
    _require_models()
    if model_name not in state["models"]:
        raise HTTPException(404, "Unknown model.")
    try:
        values = json.loads(values_json)
        if not isinstance(values, dict):
            raise ValueError
        frame = load_manual_results(
            _named_buffer(await result.read(), result.filename or "hfss.csv")
        )
        model = state["models"][model_name]
        missing = set(model.features) - set(values)
        if missing:
            raise ValueError(f"Missing features: {', '.join(sorted(missing))}")
        prediction = predict(
            model, {feature: float(values[feature]) for feature in model.features}
        )
        bundle = state["bundle"]
        if bundle.frequency_values_hz is None:
            raise ValueError("Verified dataset frequency metadata is required for comparison.")
        comparison = compare_s11_results(
            bundle.frequency_values_hz,
            np.asarray(list(prediction.values()), dtype=float),
            frame,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(400, str(exc) or "Invalid HFSS comparison request.") from exc
    state["hfss_results"] = frame
    return {"validation_status": "Compared against actual HFSS export.", **comparison}


@app.get("/api/report")
def report():
    bundle = _require_bundle()
    sections = {"Dataset": f"{len(bundle.data)} rows; {len(bundle.data.columns)} columns"}
    if state["evaluation"] is not None:
        sections["Model evaluation"] = state["evaluation"].to_string(index=False)
    return Response(
        make_pdf("Microstrip Patch Antenna Report", sections),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=antenna_report.pdf"},
    )

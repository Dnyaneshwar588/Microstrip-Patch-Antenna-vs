from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.data_loader import load_dataset
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
):
    try:
        bundle = load_dataset(
            _named_buffer(await inputs.read(), inputs.filename or "inputs.xlsx"),
            _named_buffer(await s11_real.read(), s11_real.filename or "s11_real.xlsx") if s11_real else None,
            _named_buffer(await s11_imag.read(), s11_imag.filename or "s11_imag.xlsx") if s11_imag else None,
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
    return {
        "warnings": range_warnings(values, state["prepared"].X_train),
        "prediction": predict(model, values),
    }


@app.post("/api/optimize")
def optimize(request: OptimizeRequest):
    _require_models()
    models = state["models"]
    if request.model not in models:
        raise HTTPException(404, "Unknown model.")
    model = models[request.model]
    if set(request.bounds) != set(model.features):
        raise HTTPException(400, "Bounds must be supplied for every model feature.")
    lower = [request.bounds[feature][0] for feature in model.features]
    upper = [request.bounds[feature][1] for feature in model.features]
    if any(low > high for low, high in zip(lower, upper)):
        raise HTTPException(400, "Optimization lower bounds must not exceed upper bounds.")
    result = run_algorithm(
        request.algorithm,
        make_fitness(model, model.features, (lower, upper)),
        (lower, upper),
        request.population,
        request.iterations,
        42,
    )
    serializable = {
        **result,
        "best_position": result["best_position"].tolist(),
    }
    state["optimizations"][request.algorithm] = serializable
    return serializable


@app.post("/api/hfss")
async def upload_hfss(result: Annotated[UploadFile, File(...)]):
    try:
        frame = load_manual_results(_named_buffer(await result.read(), result.filename or "hfss.csv"))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _frame_payload(frame)


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

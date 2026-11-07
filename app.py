from __future__ import annotations
import io
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.data_loader import load_dataset
from backend.data_preprocessing import prepare_data
from backend.ml_models import train_models
from backend.model_evaluation import evaluate_models
from backend.fitness_function import make_fitness
from backend.optimization_engine import run_algorithm
from backend.prediction_engine import predict, range_warnings
from backend.report_generator import make_pdf

st.set_page_config(page_title="Antenna Optimization Lab", page_icon="📡", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
[data-testid="stMetric"] {border: 1px solid #dfe7ef; padding: 1rem; border-radius: .6rem;}
</style>""", unsafe_allow_html=True)

if "bundle" not in st.session_state:
    st.session_state.update(bundle=None, models={}, evaluation=None, prepared=None, optimizations={})

with st.sidebar:
    st.title("Antenna Lab")
    page = st.radio("Navigate", ["Dashboard", "Dataset", "Machine Learning", "Prediction", "Optimization", "HFSS Validation", "Reports"])
    st.caption("Results are computed from uploaded data only. No simulated values are fabricated.")

def dataset_page():
    st.header("Dataset Management")
    st.write("Upload one combined CSV/XLSX, or the three aligned input and complex-S11 files.")
    inputs = st.file_uploader("Input parameters", type=["csv", "xlsx", "xls"], key="inputs")
    col1, col2 = st.columns(2)
    real = col1.file_uploader("S11 real (optional)", type=["csv", "xlsx", "xls"], key="real")
    imag = col2.file_uploader("S11 imaginary (optional)", type=["csv", "xlsx", "xls"], key="imag")
    if st.button("Load dataset", type="primary") and inputs:
        try:
            st.session_state.bundle = load_dataset(inputs, real, imag)
            st.session_state.models = {}
            st.session_state.evaluation = None
            st.success(f"Loaded {len(st.session_state.bundle.data):,} rows and {len(st.session_state.bundle.data.columns):,} columns.")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    bundle = st.session_state.bundle
    if bundle:
        for warning in bundle.warnings:
            st.warning(warning)
        st.dataframe(bundle.data.head(100), use_container_width=True)
        st.download_button("Download processed CSV", bundle.data.to_csv(index=False), "processed_dataset.csv", "text/csv")

def dashboard():
    st.header("Microstrip Patch Antenna Optimization")
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Upload a dataset from the Dataset page to begin.")
        return
    cols = st.columns(4)
    cols[0].metric("Samples", f"{len(bundle.data):,}")
    cols[1].metric("Numeric inputs", len(bundle.input_columns))
    cols[2].metric("Targets", len(bundle.target_columns))
    cols[3].metric("Trained models", len(st.session_state.models))
    if st.session_state.evaluation is not None:
        st.subheader("Model comparison")
        st.dataframe(st.session_state.evaluation, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(st.session_state.evaluation, x="model", y="r2", title="Validation R²"), use_container_width=True)
    else:
        st.info("Train models to populate performance metrics.")

def ml_page():
    st.header("Machine Learning")
    bundle = st.session_state.bundle
    if not bundle:
        st.info("Load a dataset first.")
        return
    numeric = [c for c in bundle.data.columns if pd.api.types.is_numeric_dtype(bundle.data[c])]
    default_targets = bundle.target_columns or [numeric[-1]]
    features = st.multiselect("Input features", numeric, default=[c for c in bundle.input_columns if c in numeric])
    targets = st.multiselect("Prediction targets", numeric, default=default_targets)
    if st.button("Train models", type="primary"):
        try:
            with st.spinner("Training and evaluating models..."):
                prepared = prepare_data(bundle.data, features, targets)
                models = train_models(prepared)
                st.session_state.prepared, st.session_state.models = prepared, models
                st.session_state.evaluation = evaluate_models(models, prepared)
            st.success("Models trained on an 80:20 holdout split.")
        except ValueError as exc:
            st.error(str(exc))
    if st.session_state.evaluation is not None:
        st.dataframe(st.session_state.evaluation, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(st.session_state.evaluation, x="model", y=["r2", "rmse", "mae"], barmode="group"), use_container_width=True)

def prediction_page():
    st.header("Performance Prediction")
    if not st.session_state.models or not st.session_state.prepared:
        st.info("Train at least one model first.")
        return
    name = st.selectbox("Model", list(st.session_state.models))
    model = st.session_state.models[name]
    values = {}
    columns = st.columns(min(4, len(model.features)))
    for index, feature in enumerate(model.features):
        series = st.session_state.prepared.X_train[feature]
        values[feature] = columns[index % len(columns)].number_input(feature, value=float(series.median()))
    if st.button("Predict", type="primary"):
        for warning in range_warnings(values, st.session_state.prepared.X_train):
            st.warning(warning)
        result = predict(model, values)
        st.dataframe(pd.DataFrame([result]), use_container_width=True)

def optimization_page():
    st.header("Antenna Optimization")
    if not st.session_state.models or not st.session_state.prepared:
        st.info("Train a model first.")
        return
    name = st.selectbox("Model", list(st.session_state.models))
    model = st.session_state.models[name]
    algorithm = st.selectbox("Algorithm", ["PSO", "DE", "GWO"])
    population = st.slider("Population", 5, 100, 25)
    iterations = st.slider("Iterations", 1, 200, 30)
    bounds = []
    st.caption("Bounds default to the observed training domain to avoid extrapolation.")
    bound_cols = st.columns(min(4, len(model.features)))
    for index, feature in enumerate(model.features):
        series = st.session_state.prepared.X_train[feature]
        lo, hi = float(series.min()), float(series.max())
        if lo == hi:
            bound_cols[index % len(bound_cols)].number_input(feature, value=lo, disabled=True)
            bounds.append((lo, hi))
        else:
            bounds.append(bound_cols[index % len(bound_cols)].slider(feature, lo, hi, (lo, hi)))
    if st.button("Run optimization", type="primary"):
        lower, upper = zip(*bounds)
        objective = make_fitness(model, model.features, (lower, upper))
        with st.spinner(f"Running {algorithm}..."):
            result = run_algorithm(algorithm, objective, (lower, upper), population, iterations, 42)
        st.session_state.optimizations[algorithm] = result
    result = st.session_state.optimizations.get(algorithm)
    if result:
        st.metric("Best fitness", f"{result['best_fitness']:.5g}")
        st.dataframe(pd.DataFrame([dict(zip(model.features, result["best_position"]))]), use_container_width=True)
        st.plotly_chart(px.line(y=result["history"], labels={"x": "Iteration", "y": "Fitness"}, title=f"{algorithm} convergence"), use_container_width=True)

def hfss_page():
    st.header("HFSS Validation")
    st.warning("Automated HFSS execution is intentionally unavailable unless PyAEDT and a licensed Ansys Electronics Desktop installation are present.")
    result = st.file_uploader("Upload actual HFSS CSV/XLSX results", type=["csv", "xlsx", "xls"])
    if result:
        try:
            from backend.hfss_integration import load_manual_results
            frame = load_manual_results(result)
            st.dataframe(frame, use_container_width=True)
            st.plotly_chart(px.line(frame), use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))

def reports_page():
    st.header("Reports")
    if not st.session_state.bundle:
        st.info("Load data and train models before generating a report.")
        return
    sections = {"Dataset": f"{len(st.session_state.bundle.data)} rows; {len(st.session_state.bundle.data.columns)} columns"}
    if st.session_state.evaluation is not None:
        sections["Model evaluation"] = st.session_state.evaluation.to_string(index=False)
    pdf = make_pdf("Microstrip Patch Antenna Report", sections)
    st.download_button("Download PDF report", pdf, "antenna_report.pdf", "application/pdf")
    st.download_button("Download dataset CSV", st.session_state.bundle.data.to_csv(index=False), "antenna_data.csv", "text/csv")

{"Dashboard": dashboard, "Dataset": dataset_page, "Machine Learning": ml_page,
 "Prediction": prediction_page, "Optimization": optimization_page,
 "HFSS Validation": hfss_page, "Reports": reports_page}[page]()

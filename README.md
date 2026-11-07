# Microstrip Patch Antenna Optimization Lab

A React/Vite research application with a FastAPI backend for loading antenna data, calculating complex S11 in dB, training regression models, and running PSO, Differential Evolution, and Grey Wolf Optimization against the trained model.

## Run the React frontend and API

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn server:app --reload
```

In a second terminal:

```powershell
npm install
npm run dev
```

Open `http://localhost:5173`. The Vite development server proxies `/api` requests to FastAPI on port 8000.

Upload either one combined CSV/XLSX or three row-aligned files: input parameters, S11 real, and S11 imaginary. The loader inspects numeric columns and does not infer undocumented frequencies. HFSS results must be actual exported results; automated execution requires an installed/licensed HFSS and PyAEDT.

The bundled METU files contain 5,000 input rows, 11 geometry/material inputs, and 201 aligned real/imaginary sweep columns. Their sweep columns are index-labelled rather than frequency-labelled, so the application calculates the complete S11 curve but does not invent a frequency range, resonant frequency, or bandwidth.

## Verified frequency metadata

The Dataset page accepts either a one-column frequency CSV/XLSX file or an explicitly verified start/stop/point-count sweep. Units are explicit (Hz, kHz, MHz, or GHz) and the number of points must equal the S11 response length. Without this metadata, resonance, bandwidth, and frequency-targeted optimization remain unavailable.

## Research workflow

1. Load aligned input, S11-real, and S11-imaginary files.
2. Optionally supply verified frequency metadata.
3. Select geometry features and complete S11 targets, then train the four surrogate regressors.
4. Predict the complete response and inspect S11, reflection coefficient, VSWR, and frequency-aware metrics.
5. Run PSO, DE, or GWO within the observed training bounds. `/api/compare` runs reproducible repeated trials and reports objective evaluations and descriptive statistics.
6. Import actual HFSS exports. `/api/hfss/compare` interpolates only over overlapping frequency points and reports prediction error; it never creates HFSS data.

HFSS automation is intentionally environment-dependent. Automated execution requires a compatible installed PyAEDT package, Ansys Electronics Desktop, a valid license, and a project whose variable names and design are configured for the requested geometry. Otherwise, export actual HFSS CSV/XLSX results and use the manual validation workflow.

## Testing

```powershell
python -m pytest -q
npm run build
```

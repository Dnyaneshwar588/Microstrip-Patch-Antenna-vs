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

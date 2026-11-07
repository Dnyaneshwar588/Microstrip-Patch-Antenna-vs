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

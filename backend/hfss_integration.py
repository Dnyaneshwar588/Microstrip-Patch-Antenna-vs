from __future__ import annotations
import pandas as pd


def load_manual_results(source) -> pd.DataFrame:
    frame = pd.read_csv(source) if str(getattr(source, "name", source)).lower().endswith(".csv") else pd.read_excel(source)
    if frame.empty:
        raise ValueError("The HFSS result file is empty.")
    return frame


def automated_hfss_status(project_path: str) -> dict[str, object]:
    try:
        from pyaedt import Hfss
    except ImportError:
        return {"available": False, "message": "PyAEDT is not installed; automated HFSS is unavailable."}
    if not project_path:
        return {"available": False, "message": "Provide an HFSS project path."}
    return {"available": True, "message": "PyAEDT is installed. Connect only from a licensed HFSS environment.",
            "project_path": project_path, "driver": Hfss}

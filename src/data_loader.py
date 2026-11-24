"""Data loading utilities for the HPGR Performance Prediction System."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_REAL_WORKBOOK: Path = PROJECT_ROOT / "data" / "real" / "HPGR_Extracted_Data.xlsx"
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "physics_assumptions.yaml"

COLUMN_METADATA: Dict[str, Dict[str, str]] = {
    "Test_ID": {"unit": "-", "role": "Identifier", "notes": "Unique pilot test ID (preserved)."},
    "Phase": {"unit": "-", "role": "Metadata", "notes": "Experimental campaign phase (Phase One / Phase Two)."},
    "Test_Description": {"unit": "-", "role": "Metadata", "notes": "Description merged from Test_Key."},
    "data_source": {"unit": "-", "role": "Provenance", "notes": "REAL for pilot tests; SYNTHETIC for exploration."},
    "roll_speed_rpm": {"unit": "rpm", "role": "Input (Collinear)", "notes": "Collinear with roll_speed_set_m_s (r=0.968)."},
    "static_gap_mm": {"unit": "mm", "role": "Constant", "notes": "Constant 9.0 mm across all 21 tests; excluded from ML."},
    "hydraulic_pressure_set_bar": {"unit": "bar", "role": "Input (Collinear)", "notes": "Setpoint hydraulic pressure."},
    "pressing_force_set_kN": {"unit": "kN", "role": "Input (Collinear)", "notes": "Collinear with actual_specific_pressure (r=1.000)."},
    "test_time_s": {"unit": "s", "role": "Metadata", "notes": "Duration of pilot test run."},
    "roll_speed_set_m_s": {"unit": "m/s", "role": "Selected Feature", "notes": "Primary roll peripheral speed feature."},
    "actual_roll_gap_mm": {"unit": "mm", "role": "Post-Process Response", "notes": "Floating operating gap; response variable."},
    "actual_specific_pressure": {"unit": "N/mm²", "role": "Selected Feature", "notes": "Primary specific pressing force feature."},
    "idle_power_kW": {"unit": "kW", "role": "Energy Component", "notes": "No-load roll drive power."},
    "power_draw_kW": {"unit": "kW", "role": "Energy Component", "notes": "Gross power draw during test."},
    "total_specific_energy_kWh_t": {"unit": "kWh/t", "role": "Alternative Energy", "notes": "Gross specific energy; kept separate from net."},
    "net_specific_energy_kWh_t": {"unit": "kWh/t", "role": "Primary Target", "notes": "Preferred net specific energy target (Pt - Pi)/W."},
    "throughput_t_h": {"unit": "t/h", "role": "Primary Target", "notes": "Press mass throughput rate."},
    "specific_throughput_constant": {"unit": "t·s/(m³·h)", "role": "Physics Diagnostic", "notes": "m-dot = W / (D × L × v)."},
    "flake_thickness_mm": {"unit": "mm", "role": "Post-Process Response", "notes": "Measured discharge flake thickness (16/21)."},
    "feed_moisture_pct": {"unit": "%", "role": "Selected Feature", "notes": "Feed moisture content (%)."},
    "feed_bulk_density_t_m3": {"unit": "t/m³", "role": "Partial Input", "notes": "6 missing rows and only 2 unique values."},
    "feed_F80_mm": {"unit": "mm", "role": "Selected Feature", "notes": "Feed 80% passing size."},
    "feed_F50_mm": {"unit": "mm", "role": "Input (Collinear)", "notes": "Collinear with feed_F80_mm (r=0.992)."},
    "centre_P80_mm": {"unit": "mm", "role": "Primary Target", "notes": "Centre product P80 (16/21 valid rows)."},
    "centre_P50_mm": {"unit": "mm", "role": "Secondary Product", "notes": "Centre product P50 (16/21 valid rows)."},
    "edge_P80_mm": {"unit": "mm", "role": "Primary Target", "notes": "Edge product P80 (16/21 valid rows)."},
    "edge_P50_mm": {"unit": "mm", "role": "Secondary Product", "notes": "Edge product P50 (16/21 valid rows)."},
    "scaleup_P80_mm": {"unit": "mm", "role": "Primary Target", "notes": "Scale-up product P80 (16/21 valid rows)."},
    "scaleup_P50_mm": {"unit": "mm", "role": "Secondary Product", "notes": "Scale-up product P50 (16/21 valid rows)."},
    "feed_mass_kg": {"unit": "kg", "role": "Mass Balance", "notes": "Feed sample mass."},
    "centre_product_mass_kg": {"unit": "kg", "role": "Mass Balance", "notes": "Centre product recovered mass."},
    "centre_product_pct": {"unit": "%", "role": "Mass Balance", "notes": "Centre share of centre+edge product."},
    "edge_product_mass_kg": {"unit": "kg", "role": "Mass Balance", "notes": "Edge product recovered mass."},
    "edge_product_pct": {"unit": "%", "role": "Mass Balance", "notes": "Edge share of centre+edge product."},
    "waste_product_mass_kg": {"unit": "kg", "role": "Mass Balance", "notes": "Waste/bypass material mass."},
    "waste_pct_of_feed": {"unit": "%", "role": "Mass Balance", "notes": "Waste percentage of feed."},
    "recovered_product_mass_kg": {"unit": "kg", "role": "Mass Balance", "notes": "Total recovered product mass."},
    "mass_reconciliation_pct": {"unit": "%", "role": "Mass Balance", "notes": "Mass balance discrepancy (%)."},
    "flake_density_t_m3": {"unit": "t/m³", "role": "Sparse Measurement", "notes": "15 missing rows (only 6 valid); excluded from ML."},
}


class WorkbookLoadError(RuntimeError):
    """Raised when the source workbook is missing or unreadable."""


def load_physics_config(config_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Load the physics and modeling assumptions YAML file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _open_excel_source(source: Union[str, Path, bytes, io.BytesIO, Any, None]) -> pd.ExcelFile:
    """Resolve a workbook path or uploaded buffer into a pandas ExcelFile."""
    if source is None:
        target = DEFAULT_REAL_WORKBOOK
        if not target.exists():
            fallback = PROJECT_ROOT / "HPGR_Extracted_Data.xlsx"
            if fallback.exists():
                target = fallback
            else:
                raise WorkbookLoadError(
                    f"Source workbook not found at '{DEFAULT_REAL_WORKBOOK}'. "
                    "Please place HPGR_Extracted_Data.xlsx in data/real/."
                )
        return pd.ExcelFile(target, engine="openpyxl")

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise WorkbookLoadError(f"Workbook path does not exist: {p}")
        return pd.ExcelFile(p, engine="openpyxl")

    if isinstance(source, bytes):
        return pd.ExcelFile(io.BytesIO(source), engine="openpyxl")

    # Streamlit UploadedFile or file-like object
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.ExcelFile(source, engine="openpyxl")


def load_workbook_sheets(source: Union[str, Path, bytes, io.BytesIO, Any, None] = None) -> Dict[str, pd.DataFrame]:
    """Load all sheets from the HPGR Excel workbook without altering source values."""
    try:
        xl = _open_excel_source(source)
    except WorkbookLoadError:
        raise
    except Exception as exc:
        raise WorkbookLoadError(f"Unable to read Excel workbook: {exc}") from exc

    if "Operating_Data" not in xl.sheet_names:
        raise WorkbookLoadError(
            f"Required sheet 'Operating_Data' is missing. Found sheets: {xl.sheet_names}"
        )

    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name in xl.sheet_names:
        sheets[sheet_name] = xl.parse(sheet_name)
    return sheets


def load_operating_data(
    source: Union[str, Path, bytes, io.BytesIO, Any, None] = None,
    include_description: bool = True,
) -> pd.DataFrame:
    """Load Operating_Data sheet, optionally merging Test_Description from Test_Key."""
    sheets = load_workbook_sheets(source)
    op_df = sheets["Operating_Data"].copy()

    if include_description and "Test_Key" in sheets:
        key_df = sheets["Test_Key"].copy()
        if "Test_ID" in key_df.columns and "Test_Description" in key_df.columns:
            desc_map = key_df[["Test_ID", "Test_Description"]].drop_duplicates(subset=["Test_ID"])
            op_df = op_df.merge(desc_map, on="Test_ID", how="left")

    op_df["data_source"] = "REAL"
    return op_df


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a complete data dictionary with units, missingness, and summary stats."""
    rows = []
    total_rows = len(df)
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        missing = int(series.isna().sum())
        missing_pct = round((missing / total_rows) * 100.0, 1) if total_rows else 0.0
        nunique = int(series.nunique(dropna=True))

        is_num = pd.api.types.is_numeric_dtype(series)
        col_min = round(float(series.min()), 4) if is_num and non_null > 0 else None
        col_max = round(float(series.max()), 4) if is_num and non_null > 0 else None
        col_mean = round(float(series.mean()), 4) if is_num and non_null > 0 else None
        col_std = round(float(series.std()), 4) if is_num and non_null > 1 else None

        meta = COLUMN_METADATA.get(col, {"unit": "-", "role": "Other", "notes": ""})
        rows.append(
            {
                "column": col,
                "unit": meta["unit"],
                "role": meta["role"],
                "dtype": str(series.dtype),
                "non_null": non_null,
                "missing": missing,
                "missing_pct": missing_pct,
                "unique_count": nunique,
                "min": col_min,
                "max": col_max,
                "mean": col_mean,
                "std": col_std,
                "notes": meta["notes"],
            }
        )
    return pd.DataFrame(rows)

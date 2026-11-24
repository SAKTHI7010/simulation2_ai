"""Preprocessing helpers that strictly preserve valid rows without fabricating targets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from src.data_loader import PROJECT_ROOT


def get_target_modeling_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Extract X, y, and identifier metadata keeping only rows where features and target exist.

    Never fabricates or imputes missing target values.
    """
    required_cols = list(features) + [target]
    valid_mask = pd.Series(True, index=df.index)
    for col in required_cols:
        valid_mask = valid_mask & df[col].notna()

    subset = df.loc[valid_mask].copy().reset_index(drop=True)
    X_df = subset[list(features)].copy()
    y_series = subset[target].astype(float).copy()
    meta_cols = [c for c in ["Test_ID", "Phase", "Test_Description", "data_source"] if c in subset.columns]
    meta_df = subset[meta_cols].copy()
    return X_df, y_series, meta_df


def compute_feature_bounds(X_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute observed min/max/summary bounds for each numeric feature in training rows."""
    bounds: Dict[str, Dict[str, float]] = {}
    for col in X_df.columns:
        s = X_df[col].dropna().astype(float)
        bounds[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "q10": float(s.quantile(0.10)),
            "q90": float(s.quantile(0.90)),
        }
    return bounds


def get_suitable_numeric_columns(
    df: pd.DataFrame,
    min_non_null: int = 12,
    min_unique: int = 3,
) -> List[str]:
    """Return numeric columns with sufficient non-missing and varying values for EDA."""
    cols: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().sum() >= min_non_null and df[col].nunique(dropna=True) >= min_unique:
                cols.append(col)
    return cols


def export_validated_dataset(
    df: pd.DataFrame,
    data_dict: pd.DataFrame,
    physics_audit_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Dict[str, Path]:
    """Export validated workbook copy, data dictionary, and physics audit to data/processed/."""
    out_dir = output_dir or (PROJECT_ROOT / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_dir / "hpgr_validated_data.xlsx"
    dict_csv_path = out_dir / "data_dictionary.csv"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Validated_Operating_Data", index=False)
        data_dict.to_excel(writer, sheet_name="Data_Dictionary", index=False)
        physics_audit_df.to_excel(writer, sheet_name="Physics_Audit", index=False)

    data_dict.to_csv(dict_csv_path, index=False, encoding="utf-8")
    return {"validated_workbook": xlsx_path, "data_dictionary_csv": dict_csv_path}

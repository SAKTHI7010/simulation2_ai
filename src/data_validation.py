"""Validation checks for the HPGR experimental workbook."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pandas as pd


class DataValidationError(ValueError):
    """Raised when required columns or structure are invalid."""


@dataclass
class ValidationReport:
    """Structured outcome of workbook validation checks."""

    is_valid: bool
    n_rows: int
    n_cols: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    target_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)
    phase_counts: Dict[str, int] = field(default_factory=dict)


def validate_operating_data(
    df: pd.DataFrame,
    required_features: Sequence[str],
    targets: Sequence[str],
    raise_on_error: bool = False,
) -> ValidationReport:
    """Validate structure, identifiers, missingness, and per-target usable row counts."""
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or df.empty:
        errors.append("Operating_Data table is empty.")
        report = ValidationReport(is_valid=False, n_rows=0, n_cols=0, errors=errors)
        if raise_on_error:
            raise DataValidationError("; ".join(errors))
        return report

    required_id_cols = ["Test_ID", "Phase"]
    missing_ids = [c for c in required_id_cols if c not in df.columns]
    if missing_ids:
        errors.append(f"Missing required identifier columns: {missing_ids}")

    missing_features = [c for c in required_features if c not in df.columns]
    if missing_features:
        errors.append(f"Missing required model feature columns: {missing_features}")

    missing_targets = [c for c in targets if c not in df.columns]
    if missing_targets:
        errors.append(f"Missing expected target columns: {missing_targets}")

    if "Test_ID" in df.columns:
        dup_ids = df["Test_ID"][df["Test_ID"].duplicated()].tolist()
        if dup_ids:
            errors.append(f"Duplicate Test_ID values detected: {dup_ids}")

    constant_cols: List[str] = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    if constant_cols:
        warnings.append(
            f"Constant or single-value columns excluded from modeling: {constant_cols}"
        )

    for feat in required_features:
        if feat in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feat]):
                errors.append(f"Feature '{feat}' must be numeric (found {df[feat].dtype}).")
            else:
                n_miss = int(df[feat].isna().sum())
                if n_miss > 0:
                    warnings.append(f"Feature '{feat}' has {n_miss} missing row(s).")

    target_availability: Dict[str, Dict[str, int]] = {}
    avail_features = [f for f in required_features if f in df.columns]
    for tgt in targets:
        if tgt not in df.columns:
            target_availability[tgt] = {"usable_rows": 0, "missing_target_rows": len(df)}
            continue
        valid_mask = df[tgt].notna()
        for f in avail_features:
            valid_mask = valid_mask & df[f].notna()
        usable = int(valid_mask.sum())
        missing_tgt = int(df[tgt].isna().sum())
        target_availability[tgt] = {
            "usable_rows": usable,
            "missing_target_rows": missing_tgt,
        }
        if usable < 10:
            warnings.append(
                f"Target '{tgt}' has only {usable} usable rows; statistical reliability is low."
            )

    phase_counts: Dict[str, int] = {}
    if "Phase" in df.columns:
        phase_counts = {str(k): int(v) for k, v in df["Phase"].value_counts(dropna=False).items()}

    if len(df) <= 30:
        warnings.append(
            f"Small pilot dataset ({len(df)} records). Exploratory LOOCV estimates only; "
            "do not treat as industrial-grade validation."
        )

    is_valid = len(errors) == 0
    report = ValidationReport(
        is_valid=is_valid,
        n_rows=len(df),
        n_cols=len(df.columns),
        errors=errors,
        warnings=warnings,
        constant_columns=constant_cols,
        target_availability=target_availability,
        phase_counts=phase_counts,
    )
    if raise_on_error and not is_valid:
        raise DataValidationError("; ".join(errors))
    return report

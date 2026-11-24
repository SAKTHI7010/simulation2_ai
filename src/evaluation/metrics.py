"""Evaluation metrics and summary reporting for HPGR regression models."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    """Compute MAE, RMSE, R2, max absolute error, and sample count.

    Uses sqrt(mean_squared_error(...)) for compatibility with scikit-learn 1.3.x.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return {
        "n_samples": int(len(y_t)),
        "mae": round(float(mean_absolute_error(y_t, y_p)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_t, y_p))), 4),
        "r2": round(float(r2_score(y_t, y_p)), 4),
        "max_abs_error": round(float(np.max(np.abs(y_t - y_p))), 4) if len(y_t) else 0.0,
    }


def build_evaluation_summary_table(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Format cross-validation evaluation metrics into a clean comparison DataFrame."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    best_idx = df.groupby("target")["rmse"].idxmin()
    df["is_best_loocv"] = False
    df.loc[best_idx, "is_best_loocv"] = True
    return df.sort_values(["target", "rmse"]).reset_index(drop=True)

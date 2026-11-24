"""Out-of-range detection, nearest-neighbor reference, and GPR uncertainty helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def check_input_ranges(
    input_dict: Dict[str, float],
    feature_bounds: Dict[str, Dict[str, float]],
    boundary_margin_pct: float = 0.05,
    feature_labels: Dict[str, str] | None = None,
    feature_units: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Check whether operating inputs fall inside, near the boundary, or outside observed bounds."""
    labels = feature_labels or {}
    units = feature_units or {}

    per_feature: Dict[str, Dict[str, Any]] = {}
    out_of_range: List[str] = []
    near_boundary: List[str] = []
    messages: List[str] = []

    for feat, bounds in feature_bounds.items():
        if feat not in input_dict:
            continue
        val = float(input_dict[feat])
        f_min = float(bounds["min"])
        f_max = float(bounds["max"])
        span = max(f_max - f_min, 1e-9)
        margin = boundary_margin_pct * span
        label = labels.get(feat, feat)
        unit = units.get(feat, "")
        unit_str = f" {unit}" if unit else ""

        if val < f_min or val > f_max:
            status = "OUTSIDE_TRAINING_RANGE"
            out_of_range.append(feat)
            direction = "below observed min" if val < f_min else "above observed max"
            messages.append(
                f"{label} ({val:.2f}{unit_str}) is {direction} [{f_min:.2f}, {f_max:.2f}]{unit_str}."
            )
        elif (val - f_min) <= margin or (f_max - val) <= margin:
            status = "NEAR_BOUNDARY"
            near_boundary.append(feat)
            messages.append(
                f"{label} ({val:.2f}{unit_str}) is near the boundary of [{f_min:.2f}, {f_max:.2f}]{unit_str}."
            )
        else:
            status = "IN_RANGE"

        norm_pos = (val - f_min) / span
        per_feature[feat] = {
            "feature": feat,
            "label": label,
            "unit": unit,
            "value": val,
            "min": f_min,
            "max": f_max,
            "status": status,
            "normalized_position": round(norm_pos, 4),
        }

    if out_of_range:
        confidence_level = "LOW_EXTRAPOLATION"
        headline = "OUTSIDE TRAINING RANGE — Low-Confidence Extrapolation"
        summary = (
            "One or more operating inputs lie outside the observed Mesaba pilot-scale range. "
            "Treat predictions as low-confidence exploratory estimates, not reliable operating recommendations."
        )
    elif near_boundary:
        confidence_level = "CAUTION_NEAR_BOUNDARY"
        headline = "NEAR TRAINING BOUNDARY — Use Engineering Caution"
        summary = (
            "Inputs are inside the observed range but close to training boundaries. "
            "With only 16-21 pilot tests, edge behavior has higher uncertainty."
        )
    else:
        confidence_level = "MODERATE_IN_RANGE"
        headline = "IN OBSERVED RANGE — Prototype Decision-Support Estimate"
        summary = (
            "All inputs fall within the observed Mesaba pilot-scale training domain. "
            "Confidence remains bounded by the small pilot dataset (16-21 tests)."
        )

    return {
        "confidence_level": confidence_level,
        "headline": headline,
        "summary": summary,
        "is_extrapolation": len(out_of_range) > 0,
        "out_of_range_features": out_of_range,
        "near_boundary_features": near_boundary,
        "messages": messages,
        "per_feature": per_feature,
    }


def compute_gpr_prediction_with_uncertainty(
    pipeline: Any,
    X_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute GPR predictive mean, std, and approximate 95% interval (mean +/- 1.96*std)."""
    preprocessor = pipeline[:-1]
    gpr_model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X_df)
    mean, std = gpr_model.predict(X_transformed, return_std=True)
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 0.0)
    ci95_low = mean - 1.96 * std
    ci95_high = mean + 1.96 * std
    return mean, std, ci95_low, ci95_high


def find_nearest_real_tests(
    input_dict: Dict[str, float],
    real_df: pd.DataFrame,
    features: Sequence[str],
    feature_bounds: Dict[str, Dict[str, float]],
    top_k: int = 3,
) -> pd.DataFrame:
    """Return top_k real pilot tests closest to the query in range-normalized feature space."""
    if real_df.empty:
        return real_df

    dists = np.zeros(len(real_df), dtype=float)
    for feat in features:
        b = feature_bounds.get(feat, {})
        span = max(float(b.get("max", 1.0)) - float(b.get("min", 0.0)), 1e-6)
        q_val = float(input_dict[feat])
        col_vals = real_df[feat].astype(float).values
        dists += ((col_vals - q_val) / span) ** 2

    dists = np.sqrt(dists / max(len(features), 1))
    out = real_df.copy()
    out["normalized_distance"] = np.round(dists, 3)
    cols = [
        "Test_ID",
        "Phase",
        "normalized_distance",
        "actual_specific_pressure",
        "roll_speed_set_m_s",
        "feed_moisture_pct",
        "feed_F80_mm",
        "throughput_t_h",
        "net_specific_energy_kWh_t",
        "scaleup_P80_mm",
        "centre_P80_mm",
        "edge_P80_mm",
    ]
    existing = [c for c in cols if c in out.columns]
    return out.sort_values("normalized_distance").head(top_k)[existing].reset_index(drop=True)

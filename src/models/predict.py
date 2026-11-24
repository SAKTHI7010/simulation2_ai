"""Inference, scenario comparison, and sensitivity sweep utilities."""

from __future__ import annotations

import json
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT, load_physics_config
from src.models.uncertainty import check_input_ranges, compute_gpr_prediction_with_uncertainty
from src.physics.hpgr_equations import (
    compute_pressing_force_from_fsp,
    compute_reduction_ratio,
    compute_rpm_from_speed,
    compute_specific_throughput_constant,
)

MODELS_DIR: Path = PROJECT_ROOT / "models"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
_ARTIFACT_CACHE: Dict[str, Any] | None = None
_META_CACHE: Dict[str, Any] | None = None
_ARTIFACT_LOCK = threading.Lock()
_FAST_CLOUD_V2 = True
_FAST_CLOUD_V3 = True

warnings.filterwarnings("ignore", message=".*sklearn\\.utils\\.parallel.*")


def load_metadata_and_reports(force_reload: bool = False) -> Dict[str, Any]:
    """Load metadata.json and LOOCV CSV reports instantaneously without loading/fitting models."""
    global _META_CACHE
    if _META_CACHE is not None and not force_reload:
        return _META_CACHE

    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        from src.models import train_models as _tm

        _tm.train_and_save_all()

    with meta_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    eval_path = PROCESSED_DIR / "model_evaluation_report.csv"
    loocv_path = PROCESSED_DIR / "loocv_predictions.csv"
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    loocv_df = pd.read_csv(loocv_path) if loocv_path.exists() else pd.DataFrame()

    _META_CACHE = {
        "metadata": metadata,
        "eval_df": eval_df,
        "loocv_df": loocv_df,
    }
    return _META_CACHE


def load_trained_artifacts(auto_train: bool = True, force_reload: bool = False) -> Dict[str, Any]:
    """Load saved metadata and fitted model pipelines, fast-refitting on version mismatch."""
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is not None and not force_reload:
        return _ARTIFACT_CACHE

    with _ARTIFACT_LOCK:
        if _ARTIFACT_CACHE is not None and not force_reload:
            return _ARTIFACT_CACHE

        import importlib
        import joblib
        import sklearn
        from src.models import train_models as _tm

        def _fast_refit(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            if not hasattr(_tm, "refit_final_pipelines_fast"):
                importlib.reload(_tm)
            return _tm.refit_final_pipelines_fast(metadata=meta, models_dir=MODELS_DIR)

        meta_path = MODELS_DIR / "metadata.json"
        if not meta_path.exists():
            if not auto_train:
                raise FileNotFoundError(f"Model metadata not found at {meta_path}")
            _tm.train_and_save_all()

        with meta_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        def _load_all_pipelines(meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
            loaded: Dict[str, Dict[str, Any]] = {}
            for tgt, tinfo in meta["targets"].items():
                loaded[tgt] = {}
                for m_key, minfo in tinfo["models"].items():
                    art_path = MODELS_DIR / minfo["artifact"]
                    if not art_path.exists():
                        raise FileNotFoundError(f"Missing model artifact: {art_path}")
                    loaded[tgt][m_key] = joblib.load(art_path)
            return loaded

        if auto_train and metadata.get("sklearn_version") != sklearn.__version__:
            pipelines = _fast_refit(metadata)
        else:
            try:
                pipelines = _load_all_pipelines(metadata)
            except Exception:
                if not auto_train:
                    raise
                pipelines = _fast_refit(metadata)

        eval_path = PROCESSED_DIR / "model_evaluation_report.csv"
        loocv_path = PROCESSED_DIR / "loocv_predictions.csv"
        eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
        loocv_df = pd.read_csv(loocv_path) if loocv_path.exists() else pd.DataFrame()

        _ARTIFACT_CACHE = {
            "metadata": metadata,
            "pipelines": pipelines,
            "eval_df": eval_df,
            "loocv_df": loocv_df,
        }
        return _ARTIFACT_CACHE


def _feature_label_unit_maps(metadata: Dict[str, Any]):
    labels = {f["name"]: f["label"] for f in metadata.get("feature_definitions", [])}
    units = {f["name"]: f["unit"] for f in metadata.get("feature_definitions", [])}
    return labels, units


def predict_all_targets(
    input_dict: Dict[str, float],
    model_key: str = "gpr",
    artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Predict all 5 feasible HPGR targets along with GPR uncertainty and range warnings."""
    bundle = artifacts or load_trained_artifacts()
    metadata = bundle["metadata"]
    pipelines = bundle["pipelines"]
    config = load_physics_config()

    features: List[str] = metadata["features"]
    X_df = pd.DataFrame([{f: float(input_dict[f]) for f in features}])
    labels, units = _feature_label_unit_maps(metadata)
    margin = float(config.get("modeling", {}).get("boundary_margin_pct", 0.05))

    global_range = check_input_ranges(
        input_dict,
        metadata["global_feature_bounds"],
        boundary_margin_pct=margin,
        feature_labels=labels,
        feature_units=units,
    )

    predictions: Dict[str, Dict[str, Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for tgt, tinfo in metadata["targets"].items():
            selected_pipe = pipelines[tgt][model_key]
            pred_val = float(selected_pipe.predict(X_df)[0])

            gpr_pipe = pipelines[tgt]["gpr"]
            gpr_mean, gpr_std, ci_low, ci_high = compute_gpr_prediction_with_uncertainty(gpr_pipe, X_df)

            tgt_range = check_input_ranges(
                input_dict,
                tinfo["feature_bounds"],
                boundary_margin_pct=margin,
                feature_labels=labels,
                feature_units=units,
            )

            # Also get all 3 models for comparison chart
            all_models: Dict[str, float] = {}
            for mk, pipe in pipelines[tgt].items():
                all_models[mk] = round(float(pipe.predict(X_df)[0]), 4)

            predictions[tgt] = {
                "target": tgt,
                "label": tinfo["label"],
                "unit": tinfo["unit"],
                "n_samples": tinfo["n_samples"],
                "model_key": model_key,
                "model_label": tinfo["models"][model_key]["label"],
                "prediction": round(pred_val, 3),
                "gpr_mean": round(float(gpr_mean[0]), 3),
                "gpr_std": round(float(gpr_std[0]), 3),
                "ci95_low": round(float(ci_low[0]), 3),
                "ci95_high": round(float(ci_high[0]), 3),
                "range_check": tgt_range,
                "all_models": all_models,
            }

    # Physics diagnostics at current operating inputs
    geom = config.get("machine_geometry", {})
    d_m = float(geom.get("roll_diameter_m", 0.750))
    w_m = float(geom.get("roll_width_m", 0.220))
    fsp = float(input_dict["actual_specific_pressure"])
    speed = float(input_dict["roll_speed_set_m_s"])
    f80 = float(input_dict["feed_F80_mm"])

    w_pred = max(predictions["throughput_t_h"]["prediction"], 1e-6)
    esp_pred = predictions["net_specific_energy_kWh_t"]["prediction"]
    p80_scaleup = max(predictions["scaleup_P80_mm"]["prediction"], 1e-6)

    physics_diagnostics = {
        "pressing_force_kN": round(compute_pressing_force_from_fsp(fsp, d_m, w_m), 1),
        "roll_speed_rpm": round(compute_rpm_from_speed(speed, d_m), 1),
        "implied_m_dot": round(compute_specific_throughput_constant(w_pred, speed, d_m, w_m), 1),
        "estimated_net_power_kW": round(esp_pred * w_pred, 2),
        "implied_reduction_ratio": round(compute_reduction_ratio(f80, p80_scaleup), 2),
    }

    return {
        "inputs": {f: float(input_dict[f]) for f in features},
        "global_range_check": global_range,
        "predictions": predictions,
        "physics_diagnostics": physics_diagnostics,
    }


def predict_scenario_comparison(
    baseline_dict: Dict[str, float],
    modified_dict: Dict[str, float],
    model_key: str = "gpr",
    artifacts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compare baseline vs modified operating conditions across all 5 targets."""
    bundle = artifacts or load_trained_artifacts()
    base_res = predict_all_targets(baseline_dict, model_key=model_key, artifacts=bundle)
    mod_res = predict_all_targets(modified_dict, model_key=model_key, artifacts=bundle)

    rows = []
    for tgt, bp in base_res["predictions"].items():
        mp = mod_res["predictions"][tgt]
        b_val = float(bp["prediction"])
        m_val = float(mp["prediction"])
        delta = m_val - b_val
        pct = (delta / b_val * 100.0) if abs(b_val) > 1e-9 else 0.0
        rows.append(
            {
                "target": tgt,
                "Target": bp["label"],
                "Unit": bp["unit"],
                "n": bp["n_samples"],
                "Baseline": round(b_val, 3),
                "Modified": round(m_val, 3),
                "Delta": round(delta, 3),
                "Delta_%": round(pct, 2),
                "Baseline_Status": bp["range_check"]["confidence_level"],
                "Modified_Status": mp["range_check"]["confidence_level"],
            }
        )

    return {
        "baseline": base_res,
        "modified": mod_res,
        "comparison_df": pd.DataFrame(rows),
    }


def generate_sensitivity_curve(
    base_input: Dict[str, float],
    sweep_feature: str,
    sweep_values: Sequence[float],
    target: str,
    model_key: str = "gpr",
    artifacts: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate 1D sensitivity sweep with GPR 95% confidence band and observed range flag."""
    bundle = artifacts or load_trained_artifacts()
    metadata = bundle["metadata"]
    features = metadata["features"]
    tinfo = metadata["targets"][target]
    bounds = tinfo["feature_bounds"][sweep_feature]
    f_min, f_max = float(bounds["min"]), float(bounds["max"])

    rows = []
    for val in sweep_values:
        pt = {f: float(base_input[f]) for f in features}
        pt[sweep_feature] = float(val)
        rows.append(pt)

    X_sweep = pd.DataFrame(rows)
    sel_pipe = bundle["pipelines"][target][model_key]
    gpr_pipe = bundle["pipelines"][target]["gpr"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        y_pred = sel_pipe.predict(X_sweep)
        gpr_mean, gpr_std, ci_low, ci_high = compute_gpr_prediction_with_uncertainty(gpr_pipe, X_sweep)

    return pd.DataFrame(
        {
            sweep_feature: np.asarray(sweep_values, dtype=float),
            "prediction": np.round(y_pred, 4),
            "gpr_mean": np.round(gpr_mean, 4),
            "gpr_std": np.round(gpr_std, 4),
            "ci95_low": np.round(ci_low, 4),
            "ci95_high": np.round(ci_high, 4),
            "in_training_range": [(f_min <= float(v) <= f_max) for v in sweep_values],
        }
    )

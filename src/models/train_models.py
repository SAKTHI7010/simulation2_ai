"""Training pipeline for HPGR Linear Regression, Random Forest, and GPR models."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    PROJECT_ROOT,
    build_data_dictionary,
    load_operating_data,
    load_physics_config,
)
from src.data_validation import validate_operating_data
from src.evaluation.metrics import build_evaluation_summary_table, compute_regression_metrics
from src.physics.feature_engineering import generate_synthetic_exploration_data
from src.physics.hpgr_equations import audit_physics_consistency
from src.preprocessing import compute_feature_bounds, export_validated_dataset, get_target_modeling_frame


def build_model_pipeline(
    model_key: str,
    random_state: int = 42,
    gpr_restarts: int = 5,
) -> Pipeline:
    """Create a scikit-learn Pipeline with leak-free preprocessing inside CV."""
    if model_key == "linear_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_key == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=4,
                        min_samples_leaf=2,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_key == "gpr":
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 1e2))
            + ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 10.0))
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GaussianProcessRegressor(
                        kernel=kernel,
                        normalize_y=True,
                        n_restarts_optimizer=gpr_restarts,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model_key: {model_key}")


def refit_final_pipelines_fast(
    metadata: Dict[str, Any] | None = None,
    workbook_path: Path | None = None,
    models_dir: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Quickly fit the 15 final target pipelines without running 90-fold LOOCV.

    Used automatically on cloud deployments when scikit-learn versions differ from
    the saved joblib pickles while preserving exact precomputed LOOCV metrics in metadata.json.
    """
    config = load_physics_config()
    df_real = load_operating_data(workbook_path)
    features = [f["name"] for f in config["selected_features"]]
    target_names = list(metadata["targets"].keys()) if metadata else [t["name"] for t in config["targets"]]
    model_keys = [m["key"] for m in config["modeling"]["models"]]
    random_state = int(config.get("modeling", {}).get("random_state", 42))
    out_models = models_dir or (PROJECT_ROOT / "models")

    pipelines: Dict[str, Dict[str, Any]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        for tgt in target_names:
            X_df, y_series, _ = get_target_modeling_frame(df_real, features, tgt)
            y_vals = y_series.values
            pipelines[tgt] = {}
            for m_key in model_keys:
                pipe = build_model_pipeline(m_key, random_state=random_state, gpr_restarts=2)
                pipe.fit(X_df, y_vals)
                pipelines[tgt][m_key] = pipe

    # Best-effort persistence on writable filesystems (non-fatal on read-only mounts)
    try:
        out_models.mkdir(parents=True, exist_ok=True)
        for tgt, m_map in pipelines.items():
            for m_key, pipe in m_map.items():
                joblib.dump(pipe, out_models / f"{tgt}__{m_key}.joblib")
        if metadata is not None:
            metadata["sklearn_version"] = sklearn.__version__
            with (out_models / "metadata.json").open("w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
    except OSError:
        pass

    return pipelines


def train_and_save_all(
    workbook_path: Path | None = None,
    models_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> Dict[str, Any]:
    """Validate real data, train LR/RF/GPR per target with LOOCV, and persist artifacts."""
    config = load_physics_config()
    df_real = load_operating_data(workbook_path)

    feature_defs = config["selected_features"]
    features = [f["name"] for f in feature_defs]
    target_defs = config["targets"]
    target_names = [t["name"] for t in target_defs]
    random_state = int(config.get("modeling", {}).get("random_state", 42))

    report = validate_operating_data(df_real, features, target_names, raise_on_error=True)

    geom = config.get("machine_geometry", {})
    d_m = float(geom.get("roll_diameter_m", 0.750))
    w_m = float(geom.get("roll_width_m", 0.220))

    data_dict = build_data_dictionary(df_real)
    physics_audit = audit_physics_consistency(df_real, diameter_m=d_m, width_m=w_m)
    out_processed = processed_dir or (PROJECT_ROOT / "data" / "processed")
    export_validated_dataset(df_real, data_dict, physics_audit, output_dir=out_processed)

    if config.get("synthetic_policy", {}).get("enabled_for_exploration", True):
        generate_synthetic_exploration_data(config)

    out_models = models_dir or (PROJECT_ROOT / "models")
    out_models.mkdir(parents=True, exist_ok=True)

    model_defs = config["modeling"]["models"]
    eval_records: List[Dict[str, Any]] = []
    loocv_rows: List[Dict[str, Any]] = []
    target_metadata: Dict[str, Any] = {}

    loo = LeaveOneOut()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        for t_def in target_defs:
            tgt = t_def["name"]
            X_df, y_series, meta_df = get_target_modeling_frame(df_real, features, tgt)
            n_samples = len(X_df)
            if n_samples < 5:
                continue

            bounds = compute_feature_bounds(X_df)
            X_vals = X_df.values
            y_vals = y_series.values

            per_model_meta: Dict[str, Any] = {}
            for m_def in model_defs:
                m_key = m_def["key"]
                m_label = m_def["label"]

                cv_pipe = build_model_pipeline(m_key, random_state=random_state)
                y_cv_pred = cross_val_predict(cv_pipe, X_vals, y_vals, cv=loo)
                metrics = compute_regression_metrics(y_vals, y_cv_pred)

                # Fit final model on all valid real rows for this target
                final_pipe = build_model_pipeline(m_key, random_state=random_state)
                final_pipe.fit(X_df, y_vals)

                artifact_name = f"{tgt}__{m_key}.joblib"
                joblib.dump(final_pipe, out_models / artifact_name)

                eval_records.append(
                    {
                        "target": tgt,
                        "target_label": t_def["label"],
                        "unit": t_def["unit"],
                        "model_key": m_key,
                        "model_label": m_label,
                        "n_samples": n_samples,
                        "validation_method": "LOOCV",
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                        "max_abs_error": metrics["max_abs_error"],
                    }
                )

                for i in range(n_samples):
                    loocv_rows.append(
                        {
                            "target": tgt,
                            "target_label": t_def["label"],
                            "unit": t_def["unit"],
                            "model_key": m_key,
                            "model_label": m_label,
                            "Test_ID": str(meta_df.iloc[i]["Test_ID"]),
                            "Phase": str(meta_df.iloc[i]["Phase"]),
                            "y_true": float(y_vals[i]),
                            "y_pred": round(float(y_cv_pred[i]), 4),
                            "residual": round(float(y_vals[i] - y_cv_pred[i]), 4),
                        }
                    )

                extra_info: Dict[str, Any] = {}
                if m_key == "linear_regression":
                    lr = final_pipe.named_steps["model"]
                    extra_info["standardized_coefficients"] = {
                        feat: round(float(c), 4) for feat, c in zip(features, lr.coef_)
                    }
                    extra_info["intercept"] = round(float(lr.intercept_), 4)
                elif m_key == "random_forest":
                    rf = final_pipe.named_steps["model"]
                    extra_info["feature_importances"] = {
                        feat: round(float(imp), 4) for feat, imp in zip(features, rf.feature_importances_)
                    }
                elif m_key == "gpr":
                    gpr = final_pipe.named_steps["model"]
                    extra_info["fitted_kernel"] = str(gpr.kernel_)

                per_model_meta[m_key] = {
                    "label": m_label,
                    "artifact": artifact_name,
                    "metrics": metrics,
                    **extra_info,
                }

            target_metadata[tgt] = {
                "label": t_def["label"],
                "unit": t_def["unit"],
                "category": t_def["category"],
                "description": t_def["description"],
                "n_samples": n_samples,
                "test_ids": meta_df["Test_ID"].tolist(),
                "feature_bounds": bounds,
                "target_min": round(float(np.min(y_vals)), 4),
                "target_max": round(float(np.max(y_vals)), 4),
                "target_mean": round(float(np.mean(y_vals)), 4),
                "models": per_model_meta,
            }

    eval_df = build_evaluation_summary_table(eval_records)
    loocv_df = pd.DataFrame(loocv_rows)

    eval_df.to_csv(out_processed / "model_evaluation_report.csv", index=False, encoding="utf-8")
    loocv_df.to_csv(out_processed / "loocv_predictions.csv", index=False, encoding="utf-8")

    metadata = {
        "project": config["project"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "data_version": "HPGR_Extracted_Data.xlsx (21 real pilot tests, Drozdiak UBC 2011)",
        "validation_method": "LOOCV",
        "features": features,
        "feature_definitions": feature_defs,
        "global_feature_bounds": compute_feature_bounds(df_real[features]),
        "dataset_rows": report.n_rows,
        "phase_counts": report.phase_counts,
        "targets": target_metadata,
        "unavailable_targets": config.get("unavailable_targets", []),
    }

    with (out_models / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return metadata


if __name__ == "__main__":
    meta = train_and_save_all()
    print(f"Trained models for {len(meta['targets'])} targets successfully.")
    for tgt, tinfo in meta["targets"].items():
        print(f"  - {tgt} (n={tinfo['n_samples']}):")
        for mk, minfo in tinfo["models"].items():
            m = minfo["metrics"]
            print(f"      {mk:18s} | MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R2={m['r2']:.4f}")

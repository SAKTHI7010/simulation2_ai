"""Streamlit data, model, and Plotly figure caching helpers for instant page transitions."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd
import streamlit as st

from src.data_loader import (
    PROJECT_ROOT,
    build_data_dictionary,
    load_operating_data,
    load_physics_config,
    load_workbook_sheets,
)
from src.data_validation import validate_operating_data
from src.models import predict as _pm
from src.physics.feature_engineering import add_physics_diagnostic_columns
from src.physics.hpgr_equations import audit_physics_consistency
from src.visualization.charts import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_force_vs_p80_multi,
    plot_missing_values_bar,
    plot_model_metrics_comparison,
    plot_residuals,
    plot_scatter_with_phase,
    plot_target_availability_bar,
)


@st.cache_data(show_spinner=False)
def cached_physics_config() -> Dict[str, Any]:
    return load_physics_config()


@st.cache_data(show_spinner=False)
def cached_metadata_and_reports() -> Dict[str, Any]:
    return _pm.load_metadata_and_reports()


@st.cache_data(show_spinner=False)
def cached_real_workbook_sheets() -> Dict[str, pd.DataFrame]:
    return load_workbook_sheets(None, include_raw_text=False)


@st.cache_data(show_spinner=False)
def cached_real_operating_data() -> pd.DataFrame:
    return load_operating_data(None)


@st.cache_data(show_spinner=False)
def cached_eda_frame() -> pd.DataFrame:
    cfg = cached_physics_config()
    geom = cfg["machine_geometry"]
    return add_physics_diagnostic_columns(
        cached_real_operating_data(),
        diameter_m=float(geom["roll_diameter_m"]),
        width_m=float(geom["roll_width_m"]),
    )


@st.cache_data(show_spinner=False)
def cached_validation_and_audit(df: pd.DataFrame, features: tuple[str, ...], targets: tuple[str, ...], d_m: float, w_m: float):
    report = validate_operating_data(df, features, targets, raise_on_error=False)
    data_dict_df = build_data_dictionary(df)
    physics_audit_df = audit_physics_consistency(df, diameter_m=d_m, width_m=w_m)
    return report, data_dict_df, physics_audit_df


@st.cache_data(show_spinner=False)
def cached_synthetic_df() -> pd.DataFrame | None:
    syn_path = PROJECT_ROOT / "data" / "synthetic" / "hpgr_synthetic_exploration.csv"
    if syn_path.exists():
        return pd.read_csv(syn_path)
    return None


@st.cache_resource(show_spinner=False)
def cached_trained_artifacts() -> Dict[str, Any]:
    return _pm.load_trained_artifacts(auto_train=True)


@st.cache_data(show_spinner=False)
def cached_missing_bar(data_dict_df: pd.DataFrame):
    return plot_missing_values_bar(data_dict_df)


@st.cache_data(show_spinner=False)
def cached_target_avail_bar(avail_df: pd.DataFrame):
    return plot_target_availability_bar(avail_df)


@st.cache_data(show_spinner=False)
def cached_scatter_phase(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    size_col: str | None = None,
    show_trendline: bool = True,
):
    return plot_scatter_with_phase(df, x_col, y_col, x_label, y_label, title, size_col, show_trendline)


@st.cache_data(show_spinner=False)
def cached_force_vs_p80(df: pd.DataFrame):
    return plot_force_vs_p80_multi(df)


@st.cache_data(show_spinner=False)
def cached_correlation_heatmap(df: pd.DataFrame, cols: tuple[str, ...]):
    return plot_correlation_heatmap(df, cols)


@st.cache_data(show_spinner=False)
def cached_model_metrics_bar(eval_df: pd.DataFrame, metric: str):
    return plot_model_metrics_comparison(eval_df, metric=metric)


@st.cache_data(show_spinner=False)
def cached_actual_vs_predicted(loocv_df: pd.DataFrame, target: str, target_label: str, unit: str, model_key: str, model_label: str):
    return plot_actual_vs_predicted(loocv_df, target, target_label, unit, model_key, model_label)


@st.cache_data(show_spinner=False)
def cached_residuals(loocv_df: pd.DataFrame, target: str, target_label: str, unit: str, model_key: str, model_label: str):
    return plot_residuals(loocv_df, target, target_label, unit, model_key, model_label)

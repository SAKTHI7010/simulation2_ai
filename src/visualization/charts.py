"""Colorful dark-mode Plotly chart builders for the HPGR Streamlit dashboard."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PHASE_COLORS: Dict[str, str] = {
    "Phase One": "#38BDF8",
    "Phase Two": "#2DD4BF",
    "Synthetic Exploration": "#FBBF24",
}

MODEL_COLORS: Dict[str, str] = {
    "linear_regression": "#38BDF8",
    "random_forest": "#FBBF24",
    "gpr": "#A78BFA",
}

MODEL_LABELS: Dict[str, str] = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gpr": "GPR",
}


def _apply_clean_layout(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=40, r=25, t=55, b=45),
        font=dict(family="sans-serif", size=12, color="#E2E8F0"),
        title_font=dict(size=15, color="#F8FAFC"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.75)",
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#111A2E",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#24324D", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#24324D", zeroline=False)
    return fig


def plot_missing_values_bar(data_dict_df: pd.DataFrame) -> go.Figure:
    """Plot horizontal bar chart of missingness across workbook columns."""
    missing_df = data_dict_df[data_dict_df["missing"] > 0].sort_values("missing", ascending=True)
    if missing_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values in dataset", showarrow=False)
        return _apply_clean_layout(fig, height=260)

    fig = px.bar(
        missing_df,
        x="missing",
        y="column",
        orientation="h",
        color="missing_pct",
        color_continuous_scale=["#38BDF8", "#A78BFA", "#FB7185"],
        text="missing",
        hover_data=["non_null", "missing_pct", "role"],
        title="Missing Value Counts by Column (Out of 21 Pilot Tests)",
        labels={"missing": "Missing Rows", "column": "Column", "missing_pct": "Missing %"},
    )
    fig.update_traces(textposition="outside")
    return _apply_clean_layout(fig, height=max(320, 32 * len(missing_df) + 80))


def plot_target_availability_bar(avail_df: pd.DataFrame) -> go.Figure:
    """Plot usable sample count per target including unavailable Bond Work Index."""
    fig = px.bar(
        avail_df,
        x="Target",
        y="Usable Rows",
        color="Status",
        text="Usable Rows",
        color_discrete_map={
            "Feasible (All 21 Tests)": "#34D399",
            "Feasible (16 Tests)": "#818CF8",
            "Unavailable (Future Work)": "#FB7185",
        },
        title="Usable Real Observations per Target (No Imputed or Fabricated Targets)",
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 25])
    return _apply_clean_layout(fig, height=360)


def plot_feature_distribution(df: pd.DataFrame, col: str, unit: str = "") -> go.Figure:
    """Histogram with marginal box plot colored by Phase."""
    label = f"{col} ({unit})" if unit and unit != "-" else col
    sub = df[[col, "Phase", "Test_ID"]].dropna(subset=[col])
    fig = px.histogram(
        sub,
        x=col,
        color="Phase",
        marginal="box",
        nbins=12,
        color_discrete_map=PHASE_COLORS,
        hover_data=["Test_ID"],
        title=f"Distribution of {label} (n={len(sub)})",
        labels={col: label},
    )
    return _apply_clean_layout(fig, height=380)


def plot_scatter_with_phase(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    size_col: str | None = None,
    show_trendline: bool = True,
) -> go.Figure:
    """Scatter plot colored by Phase with optional linear trendline and rich Test_ID hover."""
    needed = [x_col, y_col, "Phase", "Test_ID"]
    if "Test_Description" in df.columns:
        needed.append("Test_Description")
    if size_col and size_col in df.columns:
        needed.append(size_col)

    sub = df[needed].dropna(subset=[x_col, y_col]).copy()
    hover_cols = [c for c in ["Test_ID", "Phase", "Test_Description"] if c in sub.columns]

    fig = px.scatter(
        sub,
        x=x_col,
        y=y_col,
        color="Phase",
        size=size_col if size_col and size_col in sub.columns and sub[size_col].notna().all() else None,
        color_discrete_map=PHASE_COLORS,
        hover_data=hover_cols,
        title=f"{title} (n={len(sub)})",
        labels={x_col: x_label, y_col: y_label},
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=1.5, color="#0F172A"))
        if not size_col
        else dict(line=dict(width=1.2, color="#0F172A"))
    )

    if show_trendline and len(sub) >= 3 and sub[x_col].nunique() > 1:
        x_vals = sub[x_col].astype(float).values
        y_vals = sub[y_col].astype(float).values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        xs = np.linspace(x_vals.min(), x_vals.max(), 50)
        ys = slope * xs + intercept
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Linear Trend",
                line=dict(color="#FB7185", width=2.4, dash="dash"),
                hoverinfo="skip",
            )
        )

    return _apply_clean_layout(fig, height=410)


def plot_force_vs_p80_multi(df: pd.DataFrame) -> go.Figure:
    """Compare Centre P80, Edge P80, and Scale-up P80 against Specific Pressing Force."""
    sub = df[["Test_ID", "Phase", "actual_specific_pressure", "centre_P80_mm", "edge_P80_mm", "scaleup_P80_mm"]].dropna()
    melted = sub.melt(
        id_vars=["Test_ID", "Phase", "actual_specific_pressure"],
        value_vars=["centre_P80_mm", "edge_P80_mm", "scaleup_P80_mm"],
        var_name="Stream",
        value_name="P80_mm",
    )
    stream_names = {
        "centre_P80_mm": "Centre P80 (mm)",
        "edge_P80_mm": "Edge P80 (mm)",
        "scaleup_P80_mm": "Scale-up P80 (mm)",
    }
    melted["Stream"] = melted["Stream"].map(stream_names)

    fig = px.scatter(
        melted,
        x="actual_specific_pressure",
        y="P80_mm",
        color="Stream",
        symbol="Phase",
        hover_data=["Test_ID"],
        color_discrete_map={
            "Centre P80 (mm)": "#2DD4BF",
            "Edge P80 (mm)": "#FB7185",
            "Scale-up P80 (mm)": "#818CF8",
        },
        title=f"Specific Pressing Force vs. Product P80 Streams (n={len(sub)} tests with PSD)",
        labels={
            "actual_specific_pressure": "Specific Pressing Force (N/mm²)",
            "P80_mm": "Product 80% Passing Size P80 (mm)",
        },
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1.2, color="#0F172A")))
    return _apply_clean_layout(fig, height=420)


def plot_correlation_heatmap(df: pd.DataFrame, cols: Sequence[str]) -> go.Figure:
    """Annotated correlation heatmap for suitable non-constant numeric columns."""
    corr = df[list(cols)].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1.0,
        zmax=1.0,
        color_continuous_scale=["#FB7185", "#1E293B", "#38BDF8"],
        title="Pearson Correlation Matrix (Suitable Non-Constant Numeric Fields)",
        aspect="auto",
    )
    return _apply_clean_layout(fig, height=540)


def plot_model_metrics_comparison(eval_df: pd.DataFrame, metric: str = "r2") -> go.Figure:
    """Grouped bar chart comparing LR, Random Forest, and GPR across all 5 targets."""
    metric_titles = {
        "r2": "LOOCV R² Score (Higher is Better)",
        "rmse": "LOOCV RMSE (Lower is Better, Target Units)",
        "mae": "LOOCV MAE (Lower is Better, Target Units)",
    }
    df_plot = eval_df.copy()
    df_plot["Target_n"] = df_plot.apply(lambda r: f"{r['target_label']} (n={r['n_samples']})", axis=1)

    fig = px.bar(
        df_plot,
        x="Target_n",
        y=metric,
        color="model_label",
        barmode="group",
        text=metric,
        color_discrete_map={
            "Linear Regression": "#38BDF8",
            "Random Forest": "#FBBF24",
            "Gaussian Process Regression (GPR)": "#A78BFA",
        },
        title=metric_titles.get(metric, metric.upper()),
        labels={"Target_n": "Prediction Target", metric: metric.upper(), "model_label": "Model"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    return _apply_clean_layout(fig, height=410)


def plot_actual_vs_predicted(
    loocv_df: pd.DataFrame,
    target: str,
    target_label: str,
    unit: str,
    model_key: str,
    model_label: str,
) -> go.Figure:
    """Actual vs LOOCV-Predicted parity plot with 1:1 line."""
    sub = loocv_df[(loocv_df["target"] == target) & (loocv_df["model_key"] == model_key)].copy()
    if sub.empty:
        return _apply_clean_layout(go.Figure(), height=340)

    v_min = float(min(sub["y_true"].min(), sub["y_pred"].min()))
    v_max = float(max(sub["y_true"].max(), sub["y_pred"].max()))
    pad = 0.08 * max(v_max - v_min, 1e-3)
    lo, hi = v_min - pad, v_max + pad

    fig = px.scatter(
        sub,
        x="y_true",
        y="y_pred",
        color="Phase",
        hover_data=["Test_ID", "residual"],
        color_discrete_map=PHASE_COLORS,
        title=f"{model_label} — {target_label} (n={len(sub)}, LOOCV)",
        labels={"y_true": f"Actual ({unit})", "y_pred": f"LOOCV Predicted ({unit})"},
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=1.4, color="#0F172A")))

    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="1:1 Perfect Fit",
            line=dict(color="#F8FAFC", width=2, dash="dash"),
        )
    )
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi])
    return _apply_clean_layout(fig, height=360)


def plot_residuals(
    loocv_df: pd.DataFrame,
    target: str,
    target_label: str,
    unit: str,
    model_key: str,
    model_label: str,
) -> go.Figure:
    """Residual plot (Actual - Predicted) vs Test_ID."""
    sub = loocv_df[(loocv_df["target"] == target) & (loocv_df["model_key"] == model_key)].copy()
    fig = px.bar(
        sub,
        x="Test_ID",
        y="residual",
        color="Phase",
        color_discrete_map=PHASE_COLORS,
        title=f"LOOCV Residuals by Pilot Test — {model_label} ({target_label})",
        labels={"Test_ID": "Pilot Test ID", "residual": f"Residual: Actual - Pred ({unit})"},
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#F8FAFC")
    return _apply_clean_layout(fig, height=350)


def plot_prediction_model_compare(pred_info: Dict) -> go.Figure:
    """Compare all 3 models' point predictions (and GPR 95% CI) for a single target."""
    all_m = pred_info["all_models"]
    keys = ["linear_regression", "random_forest", "gpr"]
    labels = [MODEL_LABELS[k] for k in keys]
    vals = [all_m[k] for k in keys]
    colors = [MODEL_COLORS[k] for k in keys]

    err_y = [0.0, 0.0, 1.96 * pred_info["gpr_std"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
                error_y=dict(type="data", array=err_y, visible=True, color="#F8FAFC", thickness=2),
            )
        ]
    )
    fig.update_layout(
        title=f"{pred_info['label']} ({pred_info['unit']}) — All 3 Models & GPR 95% Interval",
        yaxis_title=f"{pred_info['label']} ({pred_info['unit']})",
    )
    return _apply_clean_layout(fig, height=330)


def plot_scenario_comparison_bar(comparison_df: pd.DataFrame) -> go.Figure:
    """Plot percentage delta across all 5 targets for Scenario Analysis."""
    df_p = comparison_df.copy()
    df_p["ColorGroup"] = np.where(df_p["Delta_%"] >= 0, "Increase (+)", "Decrease (-)")
    fig = px.bar(
        df_p,
        x="Target",
        y="Delta_%",
        color="ColorGroup",
        text="Delta_%",
        color_discrete_map={"Increase (+)": "#2DD4BF", "Decrease (-)": "#FB7185"},
        hover_data=["Baseline", "Modified", "Delta", "Unit"],
        title="Scenario Impact: % Change from Baseline Across All Targets",
        labels={"Delta_%": "Change (%)", "Target": "Target Output"},
    )
    fig.update_traces(texttemplate="%{text:+.2f}%", textposition="outside")
    fig.add_hline(y=0.0, line_color="#94A3B8", line_width=1.5)
    return _apply_clean_layout(fig, height=380)


def plot_sensitivity_sweep(
    sweep_df: pd.DataFrame,
    sweep_feature: str,
    x_label: str,
    y_label: str,
    min_obs: float,
    max_obs: float,
    baseline_x: float,
    modified_x: float | None = None,
) -> go.Figure:
    """Plot sensitivity curve with shaded observed range and GPR 95% uncertainty band."""
    fig = go.Figure()

    fig.add_vrect(
        x0=min_obs,
        x1=max_obs,
        fillcolor="rgba(52, 211, 153, 0.14)",
        line_width=0,
        annotation_text="Observed Pilot Range",
        annotation_position="top left",
    )

    x_arr = sweep_df[sweep_feature].values
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_arr, x_arr[::-1]]),
            y=np.concatenate([sweep_df["ci95_high"].values, sweep_df["ci95_low"].values[::-1]]),
            fill="toself",
            fillcolor="rgba(167, 139, 250, 0.24)",
            line=dict(color="rgba(255,255,255,0)"),
            name="GPR 95% Uncertainty Band (±1.96σ)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_arr,
            y=sweep_df["prediction"].values,
            mode="lines",
            name="Selected Model Prediction",
            line=dict(color="#38BDF8", width=3),
        )
    )

    fig.add_vline(
        x=baseline_x,
        line_dash="dash",
        line_color="#2DD4BF",
        annotation_text="Baseline",
        annotation_position="bottom right",
    )
    if modified_x is not None:
        fig.add_vline(
            x=modified_x,
            line_dash="dot",
            line_color="#FB7185",
            annotation_text="Modified",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"1D Sensitivity Sweep: {x_label} → {y_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return _apply_clean_layout(fig, height=410)


def plot_training_range_bullets(range_check: Dict) -> go.Figure:
    """Normalized range position chart showing where inputs sit relative to observed [0, 1]."""
    per_f = list(range_check["per_feature"].values())
    labels = [f"{item['label']} ({item['unit']})" for item in per_f]
    positions = [item["normalized_position"] * 100.0 for item in per_f]
    statuses = [item["status"] for item in per_f]
    status_colors = {
        "IN_RANGE": "#34D399",
        "NEAR_BOUNDARY": "#FBBF24",
        "OUTSIDE_TRAINING_RANGE": "#FB7185",
    }
    colors = [status_colors.get(s, "#38BDF8") for s in statuses]

    fig = go.Figure()
    fig.add_vrect(x0=0, x1=100, fillcolor="rgba(52, 211, 153, 0.14)", line_width=1, line_color="#34D399")
    fig.add_vline(x=0, line_dash="dash", line_color="#34D399", annotation_text="Obs. Min (0%)")
    fig.add_vline(x=100, line_dash="dash", line_color="#34D399", annotation_text="Obs. Max (100%)")

    fig.add_trace(
        go.Scatter(
            x=positions,
            y=labels,
            mode="markers+text",
            marker=dict(size=16, color=colors, line=dict(width=2, color="#0F172A")),
            text=[f"{item['value']:.2f} ({pos:.0f}%)" for item, pos in zip(per_f, positions)],
            textposition="top center",
            name="Current Input",
        )
    )

    x_lo = min(-25.0, min(positions) - 15.0)
    x_hi = max(125.0, max(positions) + 15.0)
    fig.update_xaxes(range=[x_lo, x_hi], title="Position Relative to Observed Training Range (%)")
    fig.update_layout(title="Input Operating Point Coverage vs. Observed Pilot Training Domain")
    return _apply_clean_layout(fig, height=340)


def plot_gpr_uncertainty_profile(
    sweep_df: pd.DataFrame,
    sweep_feature: str,
    x_label: str,
    unit: str,
    min_obs: float,
    max_obs: float,
) -> go.Figure:
    """Show how GPR predictive standard deviation expands outside the observed domain."""
    fig = go.Figure()
    fig.add_vrect(
        x0=min_obs,
        x1=max_obs,
        fillcolor="rgba(52, 211, 153, 0.14)",
        line_width=0,
        annotation_text="Observed Training Range",
        annotation_position="top left",
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_df[sweep_feature],
            y=sweep_df["gpr_std"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(251, 113, 133, 0.22)",
            line=dict(color="#FB7185", width=3),
            name="GPR Predictive Std Dev (σ)",
        )
    )
    fig.update_layout(
        title=f"GPR Predictive Uncertainty (σ) Across {x_label}",
        xaxis_title=x_label,
        yaxis_title=f"Predictive Std Dev σ ({unit})",
    )
    return _apply_clean_layout(fig, height=360)


def plot_product_psd_blend(
    psd_df: pd.DataFrame,
    centre_weight: float = 0.85,
    edge_weight: float = 0.15,
) -> go.Figure:
    """Plot log-size cumulative passing curves for Centre, Edge, and weighted Blended PSD."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=psd_df["size_mm"],
            y=psd_df["centre_passing_pct"],
            mode="lines+markers",
            name="Centre Product (Rosin-Rammler)",
            line=dict(color="#34D399", width=2.5),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=psd_df["size_mm"],
            y=psd_df["edge_passing_pct"],
            mode="lines+markers",
            name="Edge Product (Rosin-Rammler)",
            line=dict(color="#FB7185", width=2.5),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=psd_df["size_mm"],
            y=psd_df["blended_passing_pct"],
            mode="lines+markers",
            name=f"Blended ({centre_weight*100:.0f}:{edge_weight*100:.0f} Centre:Edge)",
            line=dict(color="#38BDF8", width=3.5, dash="solid"),
            marker=dict(size=7),
        )
    )
    fig.add_hline(y=80.0, line_dash="dash", line_color="#FBBF24", annotation_text="80% Passing (P80)")
    fig.update_xaxes(type="log", title="Particle Size (mm, log scale)")
    fig.update_yaxes(range=[0, 105], title="Cumulative % Passing")
    fig.update_layout(
        title=f"Synthetic Blended Product PSD ({centre_weight*100:.0f}:{edge_weight*100:.0f} Centre:Edge)",
    )
    return _apply_clean_layout(fig, height=370)

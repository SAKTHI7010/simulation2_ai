"""Page 2: Exploratory Data Analysis (EDA) for Mesaba Cu-Ni HPGR Pilot Tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

_STRETCH_KW = {"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Exploratory Analysis | HPGR", page_icon="🔍", layout="wide")

from src.data_loader import WorkbookLoadError, load_operating_data, load_physics_config
from src.physics.feature_engineering import add_physics_diagnostic_columns
from src.preprocessing import get_suitable_numeric_columns
from src.visualization.charts import (
    plot_correlation_heatmap,
    plot_force_vs_p80_multi,
    plot_scatter_with_phase,
    plot_target_availability_bar,
)
from src.visualization.ui_theme import (
    apply_custom_theme,
    render_data_provenance_sidebar,
    render_hero_banner,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="Exploratory Data Analysis",
    subtitle="Explore relationships between specific pressing force, roll speed, moisture, feed size, throughput, specific energy, and product P80.",
    badges=["INTERACTIVE PLOTLY", "PHASE-COLORED", "REAL PILOT TESTS ONLY"],
    gradient="violet-coral",
)

try:
    config = load_physics_config()
    df_raw = load_operating_data()
except WorkbookLoadError as exc:
    st.error(f"Data Load Error: {exc}")
    st.stop()

geom = config["machine_geometry"]
df = add_physics_diagnostic_columns(
    df_raw, diameter_m=float(geom["roll_diameter_m"]), width_m=float(geom["roll_width_m"])
)

ctrl1, ctrl2 = st.columns([2, 1])
with ctrl1:
    phases = sorted(df["Phase"].dropna().unique().tolist())
    sel_phases = st.multiselect("Filter Phases", phases, default=phases, key="eda_phases")
with ctrl2:
    show_trend = st.toggle("Show Linear Trendline", value=True, key="eda_trendline")

df_view = df[df["Phase"].isin(sel_phases)] if sel_phases else df.copy()
if df_view.empty:
    st.warning("Select at least one phase to display charts.")
    st.stop()

tab_rel, tab_custom, tab_corr, tab_feas = st.tabs(
    [
        "⚡ Key HPGR Relationships",
        "🎨 Custom Scatter Explorer",
        "🔥 Correlation Matrix",
        "🎯 Target Feasibility",
    ]
)

with tab_rel:
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(
            plot_scatter_with_phase(
                df_view,
                x_col="pressing_force_set_kN",
                y_col="scaleup_P80_mm",
                x_label="Pressing Force (kN)",
                y_label="Scale-up Product P80 (mm)",
                title="Pressing Force vs. Scale-up Product P80",
                show_trendline=show_trend,
            ),
            **_STRETCH_KW,
        )
    with r1c2:
        st.plotly_chart(
            plot_scatter_with_phase(
                df_view,
                x_col="roll_speed_set_m_s",
                y_col="throughput_t_h",
                x_label="Roll Peripheral Speed (m/s)",
                y_label="Press Throughput (t/h)",
                title="Roll Speed vs. Press Throughput",
                show_trendline=show_trend,
            ),
            **_STRETCH_KW,
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(
            plot_scatter_with_phase(
                df_view,
                x_col="feed_moisture_pct",
                y_col="net_specific_energy_kWh_t",
                x_label="Feed Moisture (%)",
                y_label="Net Specific Energy (kWh/t)",
                title="Feed Moisture vs. Net Specific Energy",
                show_trendline=show_trend,
            ),
            **_STRETCH_KW,
        )
    with r2c2:
        st.plotly_chart(plot_force_vs_p80_multi(df_view), **_STRETCH_KW)

with tab_custom:
    suitable_cols = get_suitable_numeric_columns(df_view, min_non_null=6, min_unique=2)
    c1, c2, c3 = st.columns(3)
    with c1:
        x_col = st.selectbox(
            "X Axis",
            suitable_cols,
            index=suitable_cols.index("actual_specific_pressure") if "actual_specific_pressure" in suitable_cols else 0,
            key="eda_custom_x",
        )
    with c2:
        y_col = st.selectbox(
            "Y Axis",
            suitable_cols,
            index=suitable_cols.index("net_specific_energy_kWh_t") if "net_specific_energy_kWh_t" in suitable_cols else 1,
            key="eda_custom_y",
        )
    with c3:
        size_choice = st.selectbox(
            "Bubble Size (optional)",
            ["None"] + suitable_cols,
            index=0,
            key="eda_custom_size",
        )
    st.plotly_chart(
        plot_scatter_with_phase(
            df_view,
            x_col=x_col,
            y_col=y_col,
            x_label=x_col,
            y_label=y_col,
            title=f"{x_col} vs. {y_col}",
            size_col=None if size_choice == "None" else size_choice,
            show_trendline=show_trend,
        ),
        **_STRETCH_KW,
    )

with tab_corr:
    corr_candidates = [
        "actual_specific_pressure",
        "roll_speed_set_m_s",
        "feed_moisture_pct",
        "feed_F80_mm",
        "actual_roll_gap_mm",
        "throughput_t_h",
        "specific_throughput_constant",
        "net_specific_energy_kWh_t",
        "total_specific_energy_kWh_t",
        "scaleup_P80_mm",
        "centre_P80_mm",
        "edge_P80_mm",
    ]
    corr_cols = [c for c in corr_candidates if c in df_view.columns and df_view[c].nunique(dropna=True) > 1]
    st.plotly_chart(plot_correlation_heatmap(df_view, corr_cols), **_STRETCH_KW)
    st.info(
        "**Feature Selection Rationale:** `pressing_force_set_kN` (`r = 1.00` with `actual_specific_pressure`), "
        "`roll_speed_rpm` (`r = 0.97` with `roll_speed_set_m_s`), and `feed_F50_mm` (`r = 0.99` with `feed_F80_mm`) "
        "are redundant collinear pairs. `static_gap_mm` is constant (`9.0 mm`), and `feed_bulk_density_t_m3` has 6 missing rows "
        "with only 2 unique values. Selecting the 4 orthogonal operating features gives 21/21 complete input rows without collinearity."
    )

with tab_feas:
    feas_rows = []
    for t_def in config["targets"]:
        tn = t_def["name"]
        n_valid = int(df[tn].notna().sum())
        feas_rows.append(
            {
                "Target": t_def["label"],
                "Column": tn,
                "Unit": t_def["unit"],
                "Usable Rows": n_valid,
                "Missing Rows": int(df[tn].isna().sum()),
                "Status": "Feasible (All 21 Tests)" if n_valid == 21 else f"Feasible ({n_valid} Tests)",
            }
        )
    feas_rows.append(
        {
            "Target": "Bond Work Index ΔBWi",
            "Column": "Not in workbook",
            "Unit": "kWh/t",
            "Usable Rows": 0,
            "Missing Rows": 21,
            "Status": "Unavailable (Future Work)",
        }
    )
    feas_df = pd.DataFrame(feas_rows)
    st.plotly_chart(plot_target_availability_bar(feas_df), **_STRETCH_KW)
    st.dataframe(feas_df, **_STRETCH_KW)

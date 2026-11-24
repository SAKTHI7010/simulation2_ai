"""Page 1: Data Overview, Validation, Physics Audit, and Real vs. Synthetic Policy."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data Overview | HPGR", page_icon="📊", layout="wide")

from src.data_loader import (
    WorkbookLoadError,
    build_data_dictionary,
    load_operating_data,
    load_physics_config,
    load_workbook_sheets,
)
from src.data_validation import validate_operating_data
from src.physics.hpgr_equations import audit_physics_consistency
from src.preprocessing import get_suitable_numeric_columns
from src.visualization.charts import (
    plot_feature_distribution,
    plot_missing_values_bar,
    plot_target_availability_bar,
)
from src.visualization.ui_theme import (
    PALETTE,
    apply_custom_theme,
    render_bwi_unavailable_card,
    render_data_provenance_sidebar,
    render_hero_banner,
    render_kpi_card,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="Data Overview & Validation Audit",
    subtitle="Inspect the 21 real Mesaba Cu-Ni pilot tests, missingness, data dictionary, physics consistency, and real vs. synthetic separation.",
    badges=["SHEET AUDIT", "NO FABRICATED TARGETS", "REAL vs SYNTHETIC SEPARATED"],
    gradient="teal-emerald",
)

config = load_physics_config()
features = [f["name"] for f in config["selected_features"]]
targets = [t["name"] for t in config["targets"]]

uploaded = st.file_uploader(
    "Optional: Upload an HPGR Excel workbook (.xlsx) to inspect and validate (defaults to built-in real dataset)",
    type=["xlsx"],
    key="data_overview_uploader",
)

try:
    source = uploaded if uploaded is not None else None
    sheets = load_workbook_sheets(source)
    df = load_operating_data(source)
    report = validate_operating_data(df, features, targets, raise_on_error=False)
except WorkbookLoadError as exc:
    st.error(f"Workbook Error: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while reading workbook: {exc}")
    st.stop()

if uploaded is not None:
    st.success(f"Loaded uploaded workbook `{uploaded.name}` successfully.")

if report.errors:
    for err in report.errors:
        st.error(f"Validation Error: {err}")
if report.warnings:
    with st.expander(f"Validation Advisories ({len(report.warnings)})", expanded=False):
        for w in report.warnings:
            st.warning(w)

data_dict_df = build_data_dictionary(df)
geom = config["machine_geometry"]
physics_audit_df = audit_physics_consistency(
    df, diameter_m=float(geom["roll_diameter_m"]), width_m=float(geom["roll_width_m"])
)

# KPI Cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Total Real Tests", f"{report.n_rows}", "rows", "Original Test_IDs preserved", PALETTE["indigo"])
with k2:
    render_kpi_card("Workbook Columns", f"{report.n_cols}", "fields", f"Constant cols: {len(report.constant_columns)}", PALETTE["sky"])
with k3:
    p1 = report.phase_counts.get("Phase One", 0)
    render_kpi_card("Phase One Tests", f"{p1}", "rows", "Primary pressure/speed/moisture", PALETTE["teal"])
with k4:
    p2 = report.phase_counts.get("Phase Two", 0)
    render_kpi_card("Phase Two Tests", f"{p2}", "rows", "2nd-stage open/closed circuit", PALETTE["violet"])
with k5:
    p80_n = report.target_availability.get("scaleup_P80_mm", {}).get("usable_rows", 0)
    render_kpi_card("P80 Valid Rows", f"{p80_n}", "/ 21", "5 Phase Two rows lack PSD", PALETTE["amber"])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Operating Data & Test Key",
        "📖 Data Dictionary & Missingness",
        "🧮 Physics Verification Audit",
        "📈 Distributions & Raw PSD",
        "🧪 Real vs. Synthetic Policy",
    ]
)

with tab1:
    phases = sorted(df["Phase"].dropna().unique().tolist())
    sel_phases = st.multiselect("Filter by Experimental Phase", phases, default=phases, key="do_phase_filter")
    filtered_df = df[df["Phase"].isin(sel_phases)] if sel_phases else df
    st.dataframe(filtered_df, use_container_width=True, height=420)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Operating Data (CSV)",
        data=csv_bytes,
        file_name="hpgr_real_operating_data.csv",
        mime="text/csv",
        key="do_download_csv",
    )
    if "Test_Key" in sheets:
        st.markdown("#### Test Key Reference")
        st.dataframe(sheets["Test_Key"], use_container_width=True, height=280)

with tab2:
    col_l, col_r = st.columns([1.2, 1.0])
    with col_l:
        st.plotly_chart(plot_missing_values_bar(data_dict_df), use_container_width=True)
    with col_r:
        avail_rows = []
        for t_def in config["targets"]:
            tn = t_def["name"]
            n_u = report.target_availability.get(tn, {}).get("usable_rows", 0)
            status = "Feasible (All 21 Tests)" if n_u == 21 else f"Feasible ({n_u} Tests)"
            avail_rows.append({"Target": t_def["label"], "Usable Rows": n_u, "Status": status})
        avail_rows.append(
            {"Target": "Bond Work Index ΔBWi", "Usable Rows": 0, "Status": "Unavailable (Future Work)"}
        )
        st.plotly_chart(plot_target_availability_bar(pd.DataFrame(avail_rows)), use_container_width=True)

    st.markdown("#### Full Data Dictionary (Units, Missingness, Roles & Range)")
    st.dataframe(data_dict_df, use_container_width=True, height=430)
    render_bwi_unavailable_card()

with tab3:
    st.info(
        f"Auditing source workbook values against UBC thesis equations with `D = {geom['roll_diameter_m']} m` "
        f"and `L = {geom['roll_width_m']} m` (`D × L = {geom['roll_diameter_m'] * geom['roll_width_m']:.3f} m²`)."
    )
    st.dataframe(physics_audit_df, use_container_width=True, height=420)
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        max_fsp_err = physics_audit_df["diff_Fsp"].abs().max()
        render_kpi_card("Max |Δ Fsp|", f"{max_fsp_err:.4f}", "N/mm²", "F / (D·L) vs reported Fsp", PALETTE["emerald"])
    with c_b:
        max_mdot_err = physics_audit_df["diff_m_dot"].abs().max()
        render_kpi_card("Max |Δ ṁ|", f"{max_mdot_err:.2f}", "t·s/m³h", "W / (D·L·v) vs reported ṁ", PALETTE["indigo"])
    with c_c:
        max_esp_err = physics_audit_df["diff_net_Esp"].abs().max()
        render_kpi_card("Max |Δ Net Esp|", f"{max_esp_err:.4f}", "kWh/t", "(Pt − Pi) / W vs reported Net Esp", PALETTE["teal"])

with tab4:
    numeric_cols = get_suitable_numeric_columns(df, min_non_null=6, min_unique=2)
    unit_map = dict(zip(data_dict_df["column"], data_dict_df["unit"]))
    sel_col = st.selectbox("Select Numeric Column to Inspect", numeric_cols, index=numeric_cols.index("throughput_t_h") if "throughput_t_h" in numeric_cols else 0, key="do_dist_col")
    st.plotly_chart(plot_feature_distribution(df, sel_col, unit_map.get(sel_col, "")), use_container_width=True)

    if "Size_Distribution_Raw" in sheets:
        with st.expander("Size_Distribution_Raw Reference Rows (53 extracted sieve rows)", expanded=False):
            st.dataframe(sheets["Size_Distribution_Raw"], use_container_width=True, height=280)
    if "Read_Me" in sheets:
        with st.expander("Workbook Read_Me Notes", expanded=False):
            st.dataframe(sheets["Read_Me"], use_container_width=True)

with tab5:
    policy_html = (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:1rem;">'
        '<div style="background:rgba(16,185,129,0.18);border:2px solid #34D399;border-radius:12px;padding:14px;">'
        '<div style="font-weight:800;color:#6EE7B7;font-size:1.05rem;">🟢 REAL Pilot Dataset (data/real/)</div>'
        '<p style="font-size:0.9rem;color:#E2E8F0;margin-top:6px;">'
        'Contains the 21 experimental Mesaba Cu-Ni sulphide pilot tests (<code>data_source = "REAL"</code>). '
        "All ML model training, LOOCV evaluation, and range bounds use <b>only</b> these real records.</p></div>"
        '<div style="background:rgba(245,158,11,0.18);border:2px solid #FBBF24;border-radius:12px;padding:14px;">'
        '<div style="font-weight:800;color:#FDE68A;font-size:1.05rem;">🟠 SYNTHETIC Exploration Data (data/synthetic/)</div>'
        '<p style="font-size:0.9rem;color:#E2E8F0;margin-top:6px;">'
        'Contains 40 physically bounded synthetic records (<code>data_source = "SYNTHETIC"</code>) generated '
        "solely for software verification and visual exploration. <b>Never</b> used for model evaluation or claimed as plant validation.</p></div></div>"
    )
    st.markdown(policy_html, unsafe_allow_html=True)
    syn_path = PROJECT_ROOT / "data" / "synthetic" / "hpgr_synthetic_exploration.csv"
    if syn_path.exists():
        syn_df = pd.read_csv(syn_path)
        st.markdown(f"#### Synthetic Exploration Table (`data_source == 'SYNTHETIC'`, {len(syn_df)} rows)")
        st.dataframe(syn_df, use_container_width=True, height=300)
    else:
        st.info("Synthetic exploration CSV has not been generated yet. Run `python -m src.models.train_models`.")

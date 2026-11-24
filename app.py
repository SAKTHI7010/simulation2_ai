"""Home page for the HPGR Performance Prediction System Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="HPGR Performance Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import WorkbookLoadError, load_operating_data, load_physics_config
from src.models.predict import load_trained_artifacts
from src.visualization.ui_theme import (
    PALETTE,
    apply_custom_theme,
    render_bwi_unavailable_card,
    render_data_provenance_sidebar,
    render_hero_banner,
    stretch_kwargs,
    render_kpi_card,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="HPGR Performance Prediction System",
    subtitle=(
        "Physics-informed machine learning decision-support simulator for pilot-scale "
        "High Pressure Grinding Rolls (Mesaba Cu-Ni sulphide ore, UBC 2011)."
    ),
    badges=[
        "REAL PILOT DATA: 21 TESTS",
        "MESABA Cu-Ni SULPHIDE ORE",
        "LR • RANDOM FOREST • GPR",
        "ROLLS: 0.750 m × 0.220 m",
        "OFFLINE PROTOTYPE",
    ],
    gradient="indigo-teal",
)

try:
    config = load_physics_config()
    df_real = load_operating_data()
    bundle = load_trained_artifacts(auto_train=True)
    metadata = bundle["metadata"]
except (WorkbookLoadError, Exception) as exc:
    st.error(f"Startup Error: {exc}")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_kpi_card("Pilot Tests", f"{len(df_real)}", "records", "Phase One (12) + Phase Two (9)", PALETTE["indigo"])
with c2:
    render_kpi_card("Core Inputs", f"{len(metadata['features'])}", "features", "0 missing across all 21 tests", PALETTE["teal"])
with c3:
    render_kpi_card("Feasible Targets", f"{len(metadata['targets'])}", "models", "Throughput, Energy & 3× P80", PALETTE["sky"])
with c4:
    render_kpi_card("ML Families", "3", "algorithms", "LR, Random Forest & GPR", PALETTE["violet"])
with c5:
    geom = config["machine_geometry"]
    render_kpi_card("Roll Geometry", f"{geom['roll_diameter_m']:.2f}×{geom['roll_width_m']:.2f}", "m", f"Static gap: {geom['static_gap_mm']:.1f} mm", PALETTE["amber"])

st.markdown("### 🔄 Physics-Informed Prediction Workflow")
workflow_html = (
    '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.25rem;">'
    '<div style="background:rgba(79,70,229,0.22);border:1px solid rgba(129,140,248,0.35);border-left:5px solid #818CF8;padding:12px;border-radius:12px;">'
    '<div style="font-weight:800;color:#C7D2FE;">1. Operating Inputs</div>'
    '<div style="font-size:0.85rem;color:#E2E8F0;margin-top:4px;">Specific Pressure (N/mm²), Roll Speed (m/s), Moisture (%), Feed F80 (mm)</div>'
    "</div>"
    '<div style="background:rgba(14,165,233,0.20);border:1px solid rgba(56,189,248,0.35);border-left:5px solid #38BDF8;padding:12px;border-radius:12px;">'
    '<div style="font-weight:800;color:#BAE6FD;">2. Physics Layer</div>'
    '<div style="font-size:0.85rem;color:#E2E8F0;margin-top:4px;">Fsp = F/(D·L), ṁ = W/(D·L·v), Net Esp = (Pt − Pi)/W</div>'
    "</div>"
    '<div style="background:rgba(16,185,129,0.20);border:1px solid rgba(52,211,153,0.35);border-left:5px solid #34D399;padding:12px;border-radius:12px;">'
    '<div style="font-weight:800;color:#A7F3D0;">3. ML Predictions</div>'
    '<div style="font-size:0.85rem;color:#E2E8F0;margin-top:4px;">Throughput (t/h), Net Energy (kWh/t), Scale-up / Centre / Edge P80 (mm)</div>'
    "</div>"
    '<div style="background:rgba(244,63,94,0.20);border:1px solid rgba(251,113,133,0.35);border-left:5px solid #FB7185;padding:12px;border-radius:12px;">'
    '<div style="font-weight:800;color:#FECDD3;">4. Confidence Guardrails</div>'
    '<div style="font-size:0.85rem;color:#E2E8F0;margin-top:4px;">Observed Range Check, GPR ±1.96σ Uncertainty & Small-Sample Warnings</div>'
    "</div></div>"
)
st.markdown(workflow_html, unsafe_allow_html=True)

st.markdown("### 🧭 Explore the Dashboard")
nav1, nav2, nav3 = st.columns(3)
with nav1:
    st.page_link("pages/1_Data_Overview.py", label="1. Data Overview & Audit", icon="📊", **stretch_kwargs())
    st.caption("Inspect the 21 pilot records, upload custom workbooks, verify physics equations, and review real vs. synthetic separation.")
    st.page_link("pages/4_Model_Comparison.py", label="4. Model Comparison (LOOCV)", icon="🏆", **stretch_kwargs())
    st.caption("Compare Linear Regression, Random Forest, and GPR with leak-free Leave-One-Out Cross-Validation metrics and parity plots.")
with nav2:
    st.page_link("pages/2_Exploratory_Analysis.py", label="2. Exploratory Analysis", icon="🔍", **stretch_kwargs())
    st.caption("Interactive scatter charts, multi-stream P80 comparison, correlation matrix, and target feasibility counts.")
    st.page_link("pages/5_Scenario_Analysis.py", label="5. Scenario Analysis (What-If)", icon="⚡", **stretch_kwargs())
    st.caption("Compare baseline vs. modified operating conditions, inspect deltas, and explore 1D sensitivity curves.")
with nav3:
    st.page_link("pages/3_Prediction.py", label="3. Interactive Prediction", icon="🎯", **stretch_kwargs())
    st.caption("Adjust operating sliders, click Predict, and view predictions across all 5 targets with GPR uncertainty.")
    st.page_link("pages/6_Confidence.py", label="6. Confidence & Limitations", icon="🛡️", **stretch_kwargs())
    st.caption("Audit training-range coverage, explore GPR uncertainty expansion, and review requirements for plant validation.")

st.markdown("### ⚠️ Scope, Provenance & Scientific Limitations")
col_a, col_b = st.columns(2)
with col_a:
    st.info(
        "**Source Material & Scope**\n\n"
        "- **Dataset:** `HPGR_Extracted_Data.xlsx` (21 pilot-scale HPGR tests from Drozdiak, UBC 2011).\n"
        "- **Ore Type:** **Mesaba copper-nickel sulphide ore** — *not* iron-ore plant data.\n"
        "- **Equipment:** Pilot HPGR (`D = 0.750 m`, `L = 0.220 m`); predictions are **not** validated for Extract Met equipment or full-scale industrial presses."
    )
with col_b:
    st.warning(
        "**Key Modeling Constraints**\n\n"
        "- **Small Sample Size:** Only 21 tests for throughput/energy and 16 tests for P80 targets; suitable for prototype comparison via LOOCV, not industrial guarantees.\n"
        "- **Centre vs. Edge P80:** Measured and modeled separately (no unverified weighting imposed).\n"
        "- **Offline Only:** No PLC/SCADA connection or automated machine control."
    )

render_bwi_unavailable_card()

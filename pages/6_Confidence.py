"""Page 6: Training Ranges, Extrapolation Guardrails, GPR Uncertainty & Future Requirements."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

_STRETCH_KW = {"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Confidence & Limitations | HPGR", page_icon="🛡️", layout="wide")

from src.models import predict as _pm
from src.models.uncertainty import check_input_ranges
from src.visualization.cache import cached_metadata_and_reports, cached_trained_artifacts
from src.visualization.charts import plot_gpr_uncertainty_profile, plot_training_range_bullets
from src.visualization.ui_theme import (
    apply_custom_theme,
    render_bwi_unavailable_card,
    render_confidence_banner,
    render_data_provenance_sidebar,
    render_hero_banner,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="Confidence, Extrapolation Guardrails & Limitations",
    subtitle="Set an operating point and press Run Confidence & Uncertainty Check to audit range coverage and GPR predictive uncertainty.",
    badges=["BUTTON-TRIGGERED AUDIT", "EXTRAPOLATION FLAGS", "GPR σ PROFILE"],
    gradient="violet-coral",
)

try:
    meta_bundle = cached_metadata_and_reports()
except Exception as exc:
    st.error(f"Error loading model metadata: {exc}")
    st.stop()

metadata = meta_bundle["metadata"]
feature_defs = metadata["feature_definitions"]
global_bounds = metadata["global_feature_bounds"]
labels = {f["name"]: f["label"] for f in feature_defs}
units = {f["name"]: f["unit"] for f in feature_defs}

base_init = st.session_state.get(
    "current_inputs", {f["name"]: float(f["default_value"]) for f in feature_defs}
)

st.markdown("### 🎛️ Audit an Operating Point Against Observed Training Ranges")
c_sliders = st.columns(4)
audit_inputs = {}
for idx, fdef in enumerate(feature_defs):
    fn = fdef["name"]
    with c_sliders[idx]:
        audit_inputs[fn] = st.slider(
            f"{fdef['label']} ({fdef['unit']})",
            min_value=float(fdef["slider_min"]),
            max_value=float(fdef["slider_max"]),
            value=float(base_init.get(fn, fdef["default_value"])),
            step=float(fdef["step"]),
            key=f"conf_slider_{fn}",
        )

run_conf_btn = st.button(
    "🛡️ Run Confidence & Uncertainty Check",
    type="primary",
    **_STRETCH_KW,
    key="run_confidence_btn",
)

if run_conf_btn:
    st.session_state["conf_result"] = check_input_ranges(
        audit_inputs,
        global_bounds,
        boundary_margin_pct=0.05,
        feature_labels=labels,
        feature_units=units,
    )
    st.session_state["conf_snapshot_inputs"] = dict(audit_inputs)

if "conf_result" not in st.session_state:
    prompt_html = (
        '<div class="hpgr-callout" style="background:rgba(167,139,250,0.18);border-color:#A78BFA;border-left:6px solid #A78BFA;">'
        '<div class="hpgr-callout-title" style="color:#DDD6FE;">🛡️ Ready for Confidence Audit</div>'
        '<div class="hpgr-callout-body">'
        "Adjust the operating sliders above and press <b>🛡️ Run Confidence & Uncertainty Check</b> to evaluate "
        "observed training-range coverage and GPR predictive uncertainty."
        "</div></div>"
    )
    st.markdown(prompt_html, unsafe_allow_html=True)
else:
    if st.session_state.get("conf_snapshot_inputs") != audit_inputs:
        st.warning("Inputs changed — press **🛡️ Run Confidence & Uncertainty Check** to refresh.")

    range_res = st.session_state["conf_result"]
    snap_inputs = st.session_state["conf_snapshot_inputs"]
    render_confidence_banner(range_res)

    col_cov, col_tbl = st.columns([1.25, 1.0])
    with col_cov:
        st.plotly_chart(plot_training_range_bullets(range_res), **_STRETCH_KW)
    with col_tbl:
        st.markdown("#### Feature Range Audit Table (All 21 Real Pilot Tests)")
        tbl_rows = []
        for item in range_res["per_feature"].values():
            tbl_rows.append(
                {
                    "Feature": f"{item['label']} ({item['unit']})",
                    "Query": round(item["value"], 3),
                    "Obs. Min": round(item["min"], 3),
                    "Obs. Max": round(item["max"], 3),
                    "Status": item["status"],
                }
            )
        st.dataframe(pd.DataFrame(tbl_rows), **_STRETCH_KW, height=280)

    with st.expander("Check Per-Target Feature Bounds (n=21 vs. n=16 P80 Subset)", expanded=False):
        tgt_rows = []
        for tgt, tinfo in metadata["targets"].items():
            for fn, b in tinfo["feature_bounds"].items():
                tgt_rows.append(
                    {
                        "Target": tinfo["label"],
                        "n_samples": tinfo["n_samples"],
                        "Feature": fn,
                        "Min": round(b["min"], 3),
                        "Max": round(b["max"], 3),
                    }
                )
        st.dataframe(pd.DataFrame(tgt_rows), **_STRETCH_KW, height=300)

    st.markdown("---")
    st.markdown("### 📉 How GPR Predictive Uncertainty (σ) Behaves Outside Training Data")
    st.info(
        "**Important Caution:** Gaussian Process Regression provides a predictive standard deviation `σ` that grows when moving "
        "away from observed training points. However, with only **16–21 pilot tests**, GPR uncertainty is **not a statistical guarantee** "
        "of true industrial error bounds — especially when extrapolating into unmeasured ore types, moisture levels, or roll geometries."
    )

    u_c1, u_c2 = st.columns(2)
    with u_c1:
        u_feat = st.selectbox(
            "Sweep Feature",
            options=[f["name"] for f in feature_defs],
            format_func=lambda fn: next(f"{f['label']} ({f['unit']})" for f in feature_defs if f["name"] == fn),
            key="conf_u_feat",
        )
    with u_c2:
        u_tgt = st.selectbox(
            "Target Uncertainty Profile",
            options=list(metadata["targets"].keys()),
            format_func=lambda k: f"{metadata['targets'][k]['label']} ({metadata['targets'][k]['unit']})",
            key="conf_u_tgt",
        )

    fdef_u = next(f for f in feature_defs if f["name"] == u_feat)
    tinfo_u = metadata["targets"][u_tgt]
    b_u = tinfo_u["feature_bounds"][u_feat]
    sweep_u_vals = np.linspace(float(fdef_u["slider_min"]), float(fdef_u["slider_max"]), 70)
    sweep_u_df = _pm.generate_sensitivity_curve(
        snap_inputs,
        sweep_feature=u_feat,
        sweep_values=sweep_u_vals,
        target=u_tgt,
        model_key="gpr",
        artifacts=cached_trained_artifacts(),
    )

    st.plotly_chart(
        plot_gpr_uncertainty_profile(
            sweep_u_df,
            sweep_feature=u_feat,
            x_label=f"{fdef_u['label']} ({fdef_u['unit']})",
            unit=tinfo_u["unit"],
            min_obs=float(b_u["min"]),
            max_obs=float(b_u["max"]),
        ),
        **_STRETCH_KW,
    )

st.markdown("---")
render_bwi_unavailable_card()

st.markdown("### 🏭 Additional Measurements Needed for Future Real-Plant Validation")
st.markdown(
    """
    | Measurement / Data Gap | Why It Is Needed for Industrial Deployment |
    |---|---|
    | **Paired Feed & HPGR Product Bond Work Index (BWi)** | Required to quantify downstream micro-cracking / grindability benefit without fabrication. |
    | **Ore Competence / SMC / Drop-Weight Parameters (`A × b`)** | Enables generalization beyond Mesaba Cu-Ni sulphide ore (e.g., to iron ore or harder copper porphyries). |
    | **Complete Feed Bulk Density & Moisture Measurements** | `feed_bulk_density_t_m3` is missing in 6 of 21 tests and only has 2 unique values in the current workbook. |
    | **Roll Surface / Stud Wear & Edge-Recirculation Measurements** | Pilot tests ran with fresh pilot rolls (`D=0.75 m, L=0.22 m`); industrial wear shifts `ṁ` and edge product split. |
    | **Calibrated Floating Roll-Gap Force Balance Data** | Needed to model `actual_roll_gap_mm` from first principles rather than treating gap solely as an observed response. |
    | **Full-Scale Plant Validation Campaigns** | Independent plant test campaigns across different roll diameters/widths and closed-circuit screen configurations. |
    """
)

"""Page 3: Interactive HPGR Performance Prediction with Explicit Predict Button."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

_STRETCH_KW = {"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Prediction | HPGR", page_icon="🎯", layout="wide")

from src.data_loader import WorkbookLoadError
from src.models import predict as _pm
from src.models.uncertainty import find_nearest_real_tests
from src.physics.hpgr_equations import (
    compute_blended_product_psd,
    compute_force_balance_gap_mm,
    compute_micro_cracking_bwi,
    compute_pressing_force_from_fsp,
    compute_rpm_from_speed,
)
from src.visualization.cache import (
    cached_metadata_and_reports,
    cached_physics_config,
    cached_real_operating_data,
    cached_trained_artifacts,
)
from src.visualization.charts import plot_prediction_model_compare, plot_product_psd_blend
from src.visualization.ui_theme import (
    PALETTE,
    apply_custom_theme,
    render_bwi_unavailable_card,
    render_confidence_banner,
    render_data_provenance_sidebar,
    render_hero_banner,
    render_kpi_card,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="Interactive HPGR Performance Prediction",
    subtitle="Set operating conditions, select a model, and press Run Prediction to estimate throughput, net specific energy, and product P80 with GPR uncertainty.",
    badges=["BUTTON-TRIGGERED PREDICTION", "UNITS ALWAYS SHOWN", "GPR ±1.96σ INTERVAL"],
    gradient="indigo-teal",
)

try:
    config = cached_physics_config()
    meta_bundle = cached_metadata_and_reports()
except (WorkbookLoadError, Exception) as exc:
    st.error(f"Error loading prediction metadata: {exc}")
    st.stop()

metadata = meta_bundle["metadata"]
feature_defs = metadata["feature_definitions"]
features = metadata["features"]
global_bounds = metadata["global_feature_bounds"]
geom = config["machine_geometry"]

if "pred_inputs" not in st.session_state:
    st.session_state["pred_inputs"] = {f["name"]: float(f["default_value"]) for f in feature_defs}

st.markdown("#### ⚡ Load a Real Pilot Test Preset (Then Click Run Prediction)")
preset_options = {
    "Custom / Median": {f["name"]: float(f["default_value"]) for f in feature_defs},
    "T1A01 (5 N/mm² High Force)": {
        "actual_specific_pressure": 4.92,
        "roll_speed_set_m_s": 0.75,
        "feed_moisture_pct": 2.5,
        "feed_F80_mm": 21.33,
    },
    "T1A04 (2 N/mm² Low Force)": {
        "actual_specific_pressure": 2.08,
        "roll_speed_set_m_s": 0.77,
        "feed_moisture_pct": 2.5,
        "feed_F80_mm": 21.33,
    },
    "T1A06 (0.90 m/s High Speed)": {
        "actual_specific_pressure": 3.94,
        "roll_speed_set_m_s": 0.90,
        "feed_moisture_pct": 2.5,
        "feed_F80_mm": 21.33,
    },
    "T1A08 (5.0% High Moisture)": {
        "actual_specific_pressure": 3.95,
        "roll_speed_set_m_s": 0.76,
        "feed_moisture_pct": 5.0,
        "feed_F80_mm": 21.33,
    },
    "T2B03 (Fine Stage-2 Dry)": {
        "actual_specific_pressure": 3.95,
        "roll_speed_set_m_s": 0.73,
        "feed_moisture_pct": 2.4,
        "feed_F80_mm": 6.16,
    },
    "Extrapolation Test (6.0 N/mm²)": {
        "actual_specific_pressure": 6.00,
        "roll_speed_set_m_s": 0.95,
        "feed_moisture_pct": 6.2,
        "feed_F80_mm": 24.0,
    },
}

p_cols = st.columns(len(preset_options))
for idx, (pname, pvals) in enumerate(preset_options.items()):
    with p_cols[idx]:
        if st.button(pname, **_STRETCH_KW, key=f"preset_btn_{idx}"):
            for fk, fv in pvals.items():
                st.session_state["pred_inputs"][fk] = float(fv)
                st.session_state[f"slider_{fk}"] = float(fv)

left_col, right_col = st.columns([1.0, 1.45])

with left_col:
    st.markdown("### 🎛️ Operating Conditions")
    current_inputs = {}
    for fdef in feature_defs:
        fname = fdef["name"]
        b = global_bounds[fname]
        cur_val = float(st.session_state["pred_inputs"].get(fname, fdef["default_value"]))
        val = st.slider(
            f"{fdef['label']} ({fdef['unit']}) — obs: [{b['min']:.2f}, {b['max']:.2f}]",
            min_value=float(fdef["slider_min"]),
            max_value=float(fdef["slider_max"]),
            value=cur_val,
            step=float(fdef["step"]),
            help=fdef["description"],
            key=f"slider_{fname}",
        )
        current_inputs[fname] = float(val)
        st.session_state["pred_inputs"][fname] = float(val)

    st.session_state["current_inputs"] = dict(current_inputs)

    model_label_map = {
        "gpr": "Gaussian Process Regression (GPR — Recommended w/ Uncertainty)",
        "linear_regression": "Linear Regression (Interpretable Baseline)",
        "random_forest": "Random Forest Regressor (Non-linear Benchmark)",
    }
    selected_model = st.radio(
        "Select Prediction Model",
        options=list(model_label_map.keys()),
        format_func=lambda k: model_label_map[k],
        index=0,
        key="pred_model_radio",
    )

    run_clicked = st.button(
        "🚀 Run Prediction",
        type="primary",
        **_STRETCH_KW,
        key="run_prediction_btn",
    )

    f_kn = compute_pressing_force_from_fsp(
        current_inputs["actual_specific_pressure"],
        float(geom["roll_diameter_m"]),
        float(geom["roll_width_m"]),
    )
    rpm = compute_rpm_from_speed(current_inputs["roll_speed_set_m_s"], float(geom["roll_diameter_m"]))
    st.info(
        f"**Equivalent Machine Setpoints (D={geom['roll_diameter_m']} m, L={geom['roll_width_m']} m):**\n"
        f"- Pressing Force: **{f_kn:.1f} kN**\n"
        f"- Roll Rotational Speed: **{rpm:.1f} rpm**"
    )

if run_clicked:
    with st.spinner("Calibrating HPGR models and computing predictions..."):
        bundle = cached_trained_artifacts()
        st.session_state["pred_result"] = _pm.predict_all_targets(
            current_inputs, model_key=selected_model, artifacts=bundle
        )
    st.session_state["pred_snapshot_inputs"] = dict(current_inputs)
    st.session_state["pred_snapshot_model"] = selected_model

with right_col:
    if "pred_result" not in st.session_state:
        prompt_html = (
            '<div class="hpgr-callout" style="background:rgba(56,189,248,0.16);border-color:#38BDF8;border-left:6px solid #38BDF8;">'
            '<div class="hpgr-callout-title" style="color:#7DD3FC;">🎯 Ready to Predict</div>'
            '<div class="hpgr-callout-body">'
            "Adjust the operating sliders or pick a pilot test preset on the left, then click "
            "<b>🚀 Run Prediction</b> to compute throughput, net specific energy, centre/edge/scale-up P80, "
            "GPR ±1.96σ intervals, and out-of-range warnings."
            "</div></div>"
        )
        st.markdown(prompt_html, unsafe_allow_html=True)
    else:
        snap_inputs = st.session_state.get("pred_snapshot_inputs", {})
        snap_model = st.session_state.get("pred_snapshot_model", "")
        if snap_inputs != current_inputs or snap_model != selected_model:
            st.warning(
                "Inputs or model selection changed since the last run — click **🚀 Run Prediction** to update results."
            )

        res = st.session_state["pred_result"]
        render_confidence_banner(res["global_range_check"])

        preds = res["predictions"]
        st.markdown("### 📊 Predicted HPGR Performance")

        m1, m2 = st.columns(2)
        tp = preds["throughput_t_h"]
        en = preds["net_specific_energy_kWh_t"]
        with m1:
            render_kpi_card(
                f"{tp['label']} (n={tp['n_samples']})",
                f"{tp['prediction']:.2f}",
                tp["unit"],
                f"GPR 95% interval: [{tp['ci95_low']:.2f}, {tp['ci95_high']:.2f}] {tp['unit']} (±{1.96*tp['gpr_std']:.2f})",
                PALETTE["indigo"],
                delta_text=f"Model: {tp['model_label']}",
                delta_color=PALETTE["indigo"],
            )
        with m2:
            render_kpi_card(
                f"{en['label']} (n={en['n_samples']})",
                f"{en['prediction']:.3f}",
                en["unit"],
                f"GPR 95% interval: [{en['ci95_low']:.3f}, {en['ci95_high']:.3f}] {en['unit']} (±{1.96*en['gpr_std']:.3f})",
                PALETTE["teal"],
                delta_text="Preferred over total specific energy",
                delta_color=PALETTE["teal"],
            )

        p1, p2, p3 = st.columns(3)
        for col_box, tgt_key, acc in [
            (p1, "scaleup_P80_mm", PALETTE["violet"]),
            (p2, "centre_P80_mm", PALETTE["emerald"]),
            (p3, "edge_P80_mm", PALETTE["coral"]),
        ]:
            info = preds[tgt_key]
            with col_box:
                render_kpi_card(
                    f"{info['label']} (n={info['n_samples']})",
                    f"{info['prediction']:.2f}",
                    info["unit"],
                    f"GPR 95% CI: [{info['ci95_low']:.2f}, {info['ci95_high']:.2f}] mm",
                    acc,
                    delta_text="Centre & Edge kept separate",
                    delta_color=acc,
                )

        diag = res["physics_diagnostics"]
        gap_cfg = config.get("synthetic_gap_and_psd", {})
        syn_gap_mm = compute_force_balance_gap_mm(
            snap_inputs["actual_specific_pressure"],
            snap_inputs["feed_moisture_pct"],
            reference_gap_mm=float(gap_cfg.get("reference_gap_mm", 22.5)),
            reference_fsp_n_mm2=float(gap_cfg.get("reference_fsp_n_mm2", 3.95)),
            stiffness_slope_mm_per_fsp=float(gap_cfg.get("stiffness_slope_mm_per_fsp", -1.45)),
            moisture_slope_mm_per_pct=float(gap_cfg.get("moisture_slope_mm_per_pct", -0.95)),
            min_gap_mm=float(gap_cfg.get("min_gap_mm", 12.0)),
            max_gap_mm=float(gap_cfg.get("max_gap_mm", 30.0)),
        )
        st.markdown("#### 🧮 Physics Diagnostics at Predicted Point")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            render_kpi_card("Specific Throughput Constant (ṁ)", f"{diag['implied_m_dot']:.1f}", "t·s/(m³·h)", "W / (D × L × v)", PALETTE["sky"])
        with d2:
            render_kpi_card("Estimated Net Grinding Power", f"{diag['estimated_net_power_kW']:.1f}", "kW", "Net Esp × Throughput", PALETTE["amber"])
        with d3:
            render_kpi_card("Scale-up Reduction Ratio", f"{diag['implied_reduction_ratio']:.2f}", "×", "Feed F80 / Scale-up P80", PALETTE["slate"])
        with d4:
            render_kpi_card("Synthetic Force-Balance Gap", f"{syn_gap_mm:.2f}", "mm", "PDF §5.3 approximation (not measured)", PALETTE["coral"])

if "pred_result" in st.session_state:
    res = st.session_state["pred_result"]
    preds = res["predictions"]
    snap_inputs = st.session_state["pred_snapshot_inputs"]

    st.markdown("---")
    st.markdown("### 🔎 Model Agreement & Nearest Real Pilot Tests")
    comp_c1, comp_c2 = st.columns([1.0, 1.35])
    with comp_c1:
        chosen_tgt = st.selectbox(
            "Target to Compare Across All 3 Models",
            options=list(preds.keys()),
            format_func=lambda k: f"{preds[k]['label']} ({preds[k]['unit']})",
            key="pred_compare_tgt",
        )
        st.plotly_chart(plot_prediction_model_compare(preds[chosen_tgt]), **_STRETCH_KW)
    with comp_c2:
        st.markdown("#### Top 3 Closest Real Pilot Tests (Range-Normalized Euclidean Distance)")
        nearest_df = find_nearest_real_tests(snap_inputs, cached_real_operating_data(), features, global_bounds, top_k=3)
        st.dataframe(nearest_df, **_STRETCH_KW, height=280)

    st.markdown("---")
    st.markdown("### 🌊 Blended Centre / Edge Product PSD & Literature BWi Micro-Cracking (PDF §5.3)")
    col_psd, col_bwi = st.columns([1.25, 1.0])

    with col_psd:
        gap_cfg = config.get("synthetic_gap_and_psd", {})
        c_pct = st.slider(
            "Centre vs. Edge Weighting (% Centre, PDF default 85:15)",
            min_value=50,
            max_value=95,
            value=int(round(float(gap_cfg.get("centre_weight", 0.85)) * 100)),
            step=5,
            key="pred_centre_weight_pct",
        )
        w_c = c_pct / 100.0
        w_e = 1.0 - w_c
        c_p80 = max(float(preds["centre_P80_mm"]["prediction"]), 0.05)
        e_p80 = max(float(preds["edge_P80_mm"]["prediction"]), 0.05)
        psd_df, blended_p80 = compute_blended_product_psd(
            c_p80,
            e_p80,
            centre_weight=w_c,
            edge_weight=w_e,
            m_centre=float(gap_cfg.get("rosin_rammler_m_centre", 0.95)),
            m_edge=float(gap_cfg.get("rosin_rammler_m_edge", 1.15)),
        )
        render_kpi_card(
            f"Synthetic {c_pct}:{100-c_pct} Blended P80",
            f"{blended_p80:.2f}",
            "mm",
            f"Centre P80={c_p80:.2f} mm ({c_pct}%) + Edge P80={e_p80:.2f} mm ({100-c_pct}%)",
            PALETTE["sky"],
        )
        st.plotly_chart(plot_product_psd_blend(psd_df, centre_weight=w_c, edge_weight=w_e), **_STRETCH_KW)

    with col_bwi:
        bwi_cfg = config.get("micro_cracking_literature", {})
        st.info(
            "**Literature-Calibrated Micro-Cracking Simulator (PDF §5.3 Ex. 5):** "
            "Estimates downstream Bond Ball Mill Work Index reduction from predicted HPGR net specific energy. "
            "*Synthetic/literature formula — not measured in `HPGR_Extracted_Data.xlsx`.*"
        )
        feed_bwi = st.number_input(
            "Feed Bond Work Index (kWh/t)",
            min_value=5.0,
            max_value=35.0,
            value=float(bwi_cfg.get("default_feed_bwi_kwh_t", 15.0)),
            step=0.5,
            key="pred_feed_bwi",
        )
        bwi_slope = st.slider(
            "Linear Micro-Cracking Slope (% BWi reduction per kWh/t)",
            min_value=1.0,
            max_value=12.0,
            value=float(bwi_cfg.get("linear_reduction_pct_per_kwh_t", 6.0)),
            step=0.5,
            key="pred_bwi_slope",
        )
        bwi_out = compute_micro_cracking_bwi(
            feed_bwi,
            preds["net_specific_energy_kWh_t"]["prediction"],
            reduction_pct_per_kwh_t=bwi_slope,
            max_reduction_pct=float(bwi_cfg.get("max_reduction_pct", 25.0)),
        )
        b1, b2 = st.columns(2)
        with b1:
            render_kpi_card(
                "Simulated Product BWi",
                f"{bwi_out['product_bwi_kwh_t']:.2f}",
                "kWh/t",
                f"From feed {bwi_out['feed_bwi_kwh_t']:.2f} kWh/t",
                PALETTE["violet"],
            )
        with b2:
            render_kpi_card(
                "Estimated BWi Reduction",
                f"-{bwi_out['bwi_delta_kwh_t']:.2f}",
                "kWh/t",
                f"{bwi_out['reduction_pct']:.1f}% micro-cracking benefit",
                PALETTE["emerald"],
            )

render_bwi_unavailable_card()

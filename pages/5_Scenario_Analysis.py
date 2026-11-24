"""Page 5: Interactive What-If Scenario Analysis & 1D Sensitivity Explorer with Predict Button."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

_STRETCH_KW = {"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Scenario Analysis | HPGR", page_icon="⚡", layout="wide")

from src.models.predict import (
    generate_sensitivity_curve,
    load_trained_artifacts,
    predict_scenario_comparison,
)
from src.visualization.charts import plot_scenario_comparison_bar, plot_sensitivity_sweep
from src.visualization.ui_theme import (
    PALETTE,
    apply_custom_theme,
    render_confidence_banner,
    render_data_provenance_sidebar,
    render_hero_banner,
    render_kpi_card,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="Scenario Analysis (What-If Explorer)",
    subtitle="Configure Baseline and Modified conditions, then press Run Scenario Prediction to inspect deltas, confidence warnings, and 1D sensitivity curves.",
    badges=["BUTTON-TRIGGERED SCENARIO", "NOT MACHINE OPTIMIZATION", "DELTA & WARNING AUDIT"],
    gradient="amber-rose",
)

try:
    bundle = load_trained_artifacts(auto_train=True)
except Exception as exc:
    st.error(f"Error loading model artifacts: {exc}")
    st.stop()

metadata = bundle["metadata"]
feature_defs = metadata["feature_definitions"]
global_bounds = metadata["global_feature_bounds"]

st.warning(
    "**Engineering Disclaimer:** Scenario comparisons illustrate prototype model sensitivity on 16–21 pilot tests "
    "of Mesaba Cu-Ni ore. They are **not** claims of validated plant optimization or safe machine control."
)

defaults = {f["name"]: float(f["default_value"]) for f in feature_defs}
base_init = dict(st.session_state["current_inputs"]) if "current_inputs" in st.session_state else dict(defaults)

if "scen_base" not in st.session_state:
    st.session_state["scen_base"] = dict(base_init)
if "scen_mod" not in st.session_state:
    mod_init = dict(base_init)
    mod_init["actual_specific_pressure"] = round(min(4.92, mod_init["actual_specific_pressure"] * 1.10), 2)
    st.session_state["scen_mod"] = mod_init

st.markdown("#### 🚀 Quick What-If Presets (Applied to Modified Scenario)")
q_cols = st.columns(6)
quick_presets = [
    ("Force +10%", lambda b: {**b, "actual_specific_pressure": round(b["actual_specific_pressure"] * 1.10, 2)}),
    ("Force -20%", lambda b: {**b, "actual_specific_pressure": round(b["actual_specific_pressure"] * 0.80, 2)}),
    ("Speed +15%", lambda b: {**b, "roll_speed_set_m_s": round(b["roll_speed_set_m_s"] * 1.15, 2)}),
    ("Moisture +2.0%", lambda b: {**b, "feed_moisture_pct": round(b["feed_moisture_pct"] + 2.0, 1)}),
    ("Finer Feed (8 mm)", lambda b: {**b, "feed_F80_mm": 8.0}),
    ("Extrapolation (6.0 N/mm²)", lambda b: {**b, "actual_specific_pressure": 6.0}),
]
for idx, (lbl, fn) in enumerate(quick_presets):
    with q_cols[idx]:
        if st.button(lbl, **_STRETCH_KW, key=f"scen_preset_{idx}"):
            new_mod = fn(st.session_state["scen_base"])
            st.session_state["scen_mod"] = new_mod
            for fk, fv in new_mod.items():
                st.session_state[f"scen_mod_{fk}"] = float(fv)

col_base, col_mod = st.columns(2)
baseline_inputs = {}
modified_inputs = {}

with col_base:
    st.markdown("### 🔵 Baseline Operating State")
    for fdef in feature_defs:
        fn = fdef["name"]
        b = global_bounds[fn]
        val = st.slider(
            f"{fdef['label']} ({fdef['unit']}) [obs: {b['min']:.2f}–{b['max']:.2f}]",
            min_value=float(fdef["slider_min"]),
            max_value=float(fdef["slider_max"]),
            value=float(st.session_state["scen_base"].get(fn, fdef["default_value"])),
            step=float(fdef["step"]),
            key=f"scen_base_{fn}",
        )
        baseline_inputs[fn] = float(val)
        st.session_state["scen_base"][fn] = float(val)

with col_mod:
    st.markdown("### 🔴 Modified Operating State")
    for fdef in feature_defs:
        fn = fdef["name"]
        b = global_bounds[fn]
        val = st.slider(
            f"{fdef['label']} ({fdef['unit']}) [obs: {b['min']:.2f}–{b['max']:.2f}]",
            min_value=float(fdef["slider_min"]),
            max_value=float(fdef["slider_max"]),
            value=float(st.session_state["scen_mod"].get(fn, fdef["default_value"])),
            step=float(fdef["step"]),
            key=f"scen_mod_{fn}",
        )
        modified_inputs[fn] = float(val)
        st.session_state["scen_mod"][fn] = float(val)

sel_model = st.radio(
    "Model for Scenario Comparison",
    options=["gpr", "linear_regression", "random_forest"],
    format_func=lambda k: {"gpr": "Gaussian Process Regression (GPR)", "linear_regression": "Linear Regression", "random_forest": "Random Forest"}[k],
    horizontal=True,
    key="scen_model_radio",
)

run_scen_btn = st.button(
    "⚡ Run Scenario Prediction",
    type="primary",
    **_STRETCH_KW,
    key="run_scenario_btn",
)

if run_scen_btn:
    st.session_state["scen_result"] = predict_scenario_comparison(
        baseline_inputs, modified_inputs, model_key=sel_model, artifacts=bundle
    )
    st.session_state["scen_snapshot"] = {
        "base": dict(baseline_inputs),
        "mod": dict(modified_inputs),
        "model": sel_model,
    }

if "scen_result" not in st.session_state:
    prompt_html = (
        '<div class="hpgr-callout" style="background:rgba(251,191,36,0.16);border-color:#FBBF24;border-left:6px solid #FBBF24;">'
        '<div class="hpgr-callout-title" style="color:#FDE68A;">⚡ Ready for Scenario Prediction</div>'
        '<div class="hpgr-callout-body">'
        "Adjust the Baseline and Modified operating states above, then press <b>⚡ Run Scenario Prediction</b> "
        "to compare predicted outputs, deltas, and sensitivity curves."
        "</div></div>"
    )
    st.markdown(prompt_html, unsafe_allow_html=True)
else:
    snap = st.session_state.get("scen_snapshot", {})
    if snap.get("base") != baseline_inputs or snap.get("mod") != modified_inputs or snap.get("model") != sel_model:
        st.warning("Scenario sliders or model changed — press **⚡ Run Scenario Prediction** to refresh results.")

    scen_out = st.session_state["scen_result"]
    comp_df = scen_out["comparison_df"]
    snap_base = snap.get("base", baseline_inputs)
    snap_mod = snap.get("mod", modified_inputs)
    snap_model = snap.get("model", sel_model)

    w_c1, w_c2 = st.columns(2)
    with w_c1:
        st.markdown("**Baseline Range Status**")
        render_confidence_banner(scen_out["baseline"]["global_range_check"])
    with w_c2:
        st.markdown("**Modified Range Status**")
        render_confidence_banner(scen_out["modified"]["global_range_check"])

    st.markdown("### 📈 Predicted Output Deltas (Modified vs. Baseline)")
    k_cols = st.columns(len(comp_df))
    accents = [PALETTE["indigo"], PALETTE["teal"], PALETTE["violet"], PALETTE["emerald"], PALETTE["coral"]]
    for idx, (_, row) in enumerate(comp_df.iterrows()):
        with k_cols[idx]:
            d_col = "#34D399" if row["Delta_%"] >= 0 else "#FB7185"
            render_kpi_card(
                f"{row['Target']} (n={row['n']})",
                f"{row['Modified']:.2f}",
                row["Unit"],
                f"Baseline: {row['Baseline']:.2f} {row['Unit']}",
                accents[idx % len(accents)],
                delta_text=f"Δ {row['Delta']:+.3f} ({row['Delta_%']:+.2f}%)",
                delta_color=d_col,
            )

    st.plotly_chart(plot_scenario_comparison_bar(comp_df), **_STRETCH_KW)
    st.dataframe(comp_df.drop(columns=["target"]), **_STRETCH_KW)

    st.markdown("---")
    st.markdown("### 🌊 Interactive 1D Sensitivity Curve (Around Baseline)")
    s_c1, s_c2 = st.columns(2)
    with s_c1:
        sweep_feat = st.selectbox(
            "Feature to Sweep",
            options=[f["name"] for f in feature_defs],
            format_func=lambda fn: next(f"{f['label']} ({f['unit']})" for f in feature_defs if f["name"] == fn),
            key="scen_sweep_feat",
        )
    with s_c2:
        sweep_tgt = st.selectbox(
            "Target Response",
            options=list(metadata["targets"].keys()),
            format_func=lambda k: f"{metadata['targets'][k]['label']} ({metadata['targets'][k]['unit']})",
            key="scen_sweep_tgt",
        )

    fdef_sel = next(f for f in feature_defs if f["name"] == sweep_feat)
    tinfo_sel = metadata["targets"][sweep_tgt]
    b_sel = tinfo_sel["feature_bounds"][sweep_feat]
    sweep_vals = np.linspace(float(fdef_sel["slider_min"]), float(fdef_sel["slider_max"]), 60)

    sweep_df = generate_sensitivity_curve(
        snap_base,
        sweep_feature=sweep_feat,
        sweep_values=sweep_vals,
        target=sweep_tgt,
        model_key=snap_model,
        artifacts=bundle,
    )

    st.plotly_chart(
        plot_sensitivity_sweep(
            sweep_df,
            sweep_feature=sweep_feat,
            x_label=f"{fdef_sel['label']} ({fdef_sel['unit']})",
            y_label=f"{tinfo_sel['label']} ({tinfo_sel['unit']})",
            min_obs=float(b_sel["min"]),
            max_obs=float(b_sel["max"]),
            baseline_x=snap_base[sweep_feat],
            modified_x=snap_mod[sweep_feat],
        ),
        **_STRETCH_KW,
    )

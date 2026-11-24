"""Page 4: Model Comparison (LOOCV Metrics, Actual-vs-Predicted, and Feature Effects)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

_STRETCH_KW = {"width": "stretch"} if tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Model Comparison | HPGR", page_icon="🏆", layout="wide")

from src.models.predict import load_trained_artifacts
from src.visualization.charts import (
    plot_actual_vs_predicted,
    plot_model_metrics_comparison,
    plot_residuals,
)
from src.visualization.ui_theme import (
    PALETTE,
    apply_custom_theme,
    render_data_provenance_sidebar,
    render_hero_banner,
    render_kpi_card,
)

apply_custom_theme()
render_data_provenance_sidebar()

render_hero_banner(
    title="Model Comparison & LOOCV Evaluation",
    subtitle="Leak-free Leave-One-Out Cross-Validation (LOOCV) comparing Linear Regression, Random Forest, and Gaussian Process Regression on identical real rows.",
    badges=["ZERO DATA LEAKAGE", "IDENTICAL LOOCV SPLITS", "ACTUAL vs PREDICTED"],
    gradient="indigo-teal",
)

try:
    bundle = load_trained_artifacts(auto_train=True)
except Exception as exc:
    st.error(f"Error loading evaluation artifacts: {exc}")
    st.stop()

metadata = bundle["metadata"]
eval_df = bundle["eval_df"]
loocv_df = bundle["loocv_df"]

if eval_df.empty or loocv_df.empty:
    st.warning("Evaluation tables are empty. Run `python -m src.models.train_models`.")
    st.stop()

st.info(
    "**Validation Protocol:** All preprocessing (imputation & scaling) runs inside each scikit-learn `Pipeline` fold. "
    "Models are evaluated via **Leave-One-Out Cross-Validation (LOOCV)** on the exact same usable rows (`n=21` for throughput and net specific energy; "
    "`n=16` for scale-up, centre, and edge P80). Models are **never** compared by training score."
)

# Summary KPIs
best_rows = eval_df[eval_df["is_best_loocv"] == True]  # noqa: E712
cols = st.columns(len(best_rows))
accents = [PALETTE["indigo"], PALETTE["teal"], PALETTE["violet"], PALETTE["emerald"], PALETTE["coral"]]
for idx, (_, row) in enumerate(best_rows.iterrows()):
    with cols[idx]:
        render_kpi_card(
            f"{row['target_label']} (n={row['n_samples']})",
            f"R² = {row['r2']:.3f}",
            "",
            f"Best: {row['model_label']} | RMSE={row['rmse']:.3f} {row['unit']}",
            accents[idx % len(accents)],
        )

tab_table, tab_parity, tab_effects = st.tabs(
    [
        "📊 LOOCV Metrics & Benchmarks",
        "🎯 Actual vs. Predicted & Residuals",
        "🔬 Feature Effects & Model Insights",
    ]
)

with tab_table:
    metric_choice = st.radio(
        "Select Metric to Plot Across Targets",
        options=["r2", "rmse", "mae"],
        format_func=lambda m: {"r2": "R² (Higher is Better)", "rmse": "RMSE (Lower is Better)", "mae": "MAE (Lower is Better)"}[m],
        horizontal=True,
        key="mc_metric_radio",
    )
    st.plotly_chart(plot_model_metrics_comparison(eval_df, metric=metric_choice), **_STRETCH_KW)

    st.markdown("#### Complete Per-Target Cross-Validation Table")
    display_cols = [
        "target_label",
        "unit",
        "n_samples",
        "validation_method",
        "model_label",
        "mae",
        "rmse",
        "r2",
        "max_abs_error",
        "is_best_loocv",
    ]
    st.dataframe(eval_df[display_cols], **_STRETCH_KW, height=420)

with tab_parity:
    tgt_options = list(metadata["targets"].keys())
    sel_tgt = st.selectbox(
        "Select Target to Inspect Parity & Residuals",
        options=tgt_options,
        format_func=lambda k: f"{metadata['targets'][k]['label']} ({metadata['targets'][k]['unit']}, n={metadata['targets'][k]['n_samples']})",
        key="mc_parity_tgt",
    )
    tinfo = metadata["targets"][sel_tgt]

    p_c1, p_c2, p_c3 = st.columns(3)
    for col_box, m_key, m_lbl in [
        (p_c1, "linear_regression", "Linear Regression"),
        (p_c2, "random_forest", "Random Forest"),
        (p_c3, "gpr", "Gaussian Process Regression (GPR)"),
    ]:
        with col_box:
            st.plotly_chart(
                plot_actual_vs_predicted(loocv_df, sel_tgt, tinfo["label"], tinfo["unit"], m_key, m_lbl),
                **_STRETCH_KW,
            )

    res_model = st.selectbox(
        "Select Model for Residual Breakdown by Test_ID",
        options=["gpr", "linear_regression", "random_forest"],
        format_func=lambda k: tinfo["models"][k]["label"],
        key="mc_resid_model",
    )
    st.plotly_chart(
        plot_residuals(loocv_df, sel_tgt, tinfo["label"], tinfo["unit"], res_model, tinfo["models"][res_model]["label"]),
        **_STRETCH_KW,
    )

with tab_effects:
    sel_eff_tgt = st.selectbox(
        "Select Target for Feature Importance & Coefficients",
        options=list(metadata["targets"].keys()),
        format_func=lambda k: f"{metadata['targets'][k]['label']} ({metadata['targets'][k]['unit']})",
        key="mc_eff_tgt",
    )
    tinfo_e = metadata["targets"][sel_eff_tgt]
    e_c1, e_c2 = st.columns(2)

    with e_c1:
        lr_coefs = tinfo_e["models"]["linear_regression"].get("standardized_coefficients", {})
        coef_df = pd.DataFrame(
            [{"Feature": k, "Standardized Coef": v} for k, v in lr_coefs.items()]
        )
        fig_lr = px.bar(
            coef_df,
            x="Feature",
            y="Standardized Coef",
            color="Standardized Coef",
            color_continuous_scale=["#FB7185", "#1E293B", "#38BDF8"],
            text="Standardized Coef",
            title=f"Linear Regression Standardized Coefficients — {tinfo_e['label']}",
        )
        fig_lr.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_lr.update_layout(
            template="plotly_dark",
            height=370,
            plot_bgcolor="#0F172A",
            paper_bgcolor="#111A2E",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig_lr, **_STRETCH_KW)

    with e_c2:
        rf_imps = tinfo_e["models"]["random_forest"].get("feature_importances", {})
        imp_df = pd.DataFrame(
            [{"Feature": k, "Importance": v} for k, v in rf_imps.items()]
        ).sort_values("Importance", ascending=True)
        fig_rf = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=["#38BDF8", "#A78BFA", "#FBBF24"],
            text="Importance",
            title=f"Random Forest Feature Importances — {tinfo_e['label']}",
        )
        fig_rf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_rf.update_layout(
            template="plotly_dark",
            height=370,
            plot_bgcolor="#0F172A",
            paper_bgcolor="#111A2E",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig_rf, **_STRETCH_KW)

    st.info(
        f"**Fitted GPR Kernel (`{tinfo_e['label']}`):** `{tinfo_e['models']['gpr'].get('fitted_kernel', '')}`\n\n"
        "**Why GPR (Linear + RBF) & Linear Regression outperform Random Forest on this dataset:** "
        "Decision trees cannot extrapolate beyond observed leaf averages when a single extreme test (e.g., `T1A05` at 0.61 m/s or `T1A06` at 0.90 m/s) "
        "is held out in LOOCV, whereas Linear Regression and the Linear+RBF GPR kernel preserve smooth physical trends across speed and force."
    )

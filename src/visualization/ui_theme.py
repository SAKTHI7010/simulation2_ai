"""Colorful dark-mode UI theme, hero banners, KPI cards, and provenance badges for Streamlit 1.35."""

from __future__ import annotations

from html import escape
from typing import Any, Dict, Sequence

import streamlit as st

PALETTE: Dict[str, str] = {
    "indigo": "#818CF8",
    "teal": "#2DD4BF",
    "coral": "#FB7185",
    "amber": "#FBBF24",
    "violet": "#A78BFA",
    "sky": "#38BDF8",
    "emerald": "#34D399",
    "slate": "#94A3B8",
    "muted": "#94A3B8",
}

GRADIENTS: Dict[str, str] = {
    "indigo-teal": "linear-gradient(135deg, #4338CA 0%, #0284C7 52%, #0D9488 100%)",
    "violet-coral": "linear-gradient(135deg, #6D28D9 0%, #DB2777 55%, #EA580C 100%)",
    "teal-emerald": "linear-gradient(135deg, #0F766E 0%, #0284C7 55%, #4F46E5 100%)",
    "amber-rose": "linear-gradient(135deg, #D97706 0%, #E11D48 55%, #7C3AED 100%)",
}


def apply_custom_theme() -> None:
    """Inject vibrant dark-mode CSS styling compatible with Streamlit 1.35.0."""
    st.markdown(
        """<style>
.stApp {
    background: linear-gradient(145deg, #090E1F 0%, #111936 50%, #0B1F2E 100%);
    color: #F1F5F9;
}
[data-testid="stHeader"] {
    background: rgba(9, 14, 31, 0.82) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stToolbar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12103A 0%, #1E1B5E 55%, #0D3B4C 100%) !important;
    color: #F8FAFC;
    border-right: 1px solid rgba(56, 189, 248, 0.22);
}
[data-testid="stSidebarNav"] {
    background: transparent !important;
}
[data-testid="stSidebarNav"]::before,
[data-testid="stSidebarNav"]::after,
[data-testid="stSidebarNav"] > div {
    background: transparent !important;
    background-image: none !important;
}
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] code,
.stApp code {
    background: rgba(56, 189, 248, 0.16) !important;
    color: #E0F2FE !important;
    border: 1px solid rgba(56, 189, 248, 0.30);
    border-radius: 6px;
    padding: 0.1rem 0.35rem;
}
.hpgr-hero {
    padding: 1.45rem 1.75rem;
    border-radius: 16px;
    color: #FFFFFF;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 12px 28px -6px rgba(14, 165, 233, 0.32);
}
.hpgr-hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #FFFFFF !important;
}
.hpgr-hero p {
    margin: 0;
    font-size: 1.0rem;
    opacity: 0.96;
    color: #F8FAFC !important;
}
.hpgr-badge-row {
    margin-top: 0.85rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.hpgr-pill {
    display: inline-block;
    padding: 0.22rem 0.72rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    background: rgba(15, 23, 42, 0.36);
    border: 1px solid rgba(255, 255, 255, 0.35);
    color: #FFFFFF;
}
.hpgr-kpi {
    background: linear-gradient(145deg, rgba(23, 33, 59, 0.95), rgba(15, 23, 42, 0.96));
    border-radius: 14px;
    padding: 1rem 1.15rem;
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: 0 8px 20px rgba(2, 6, 23, 0.45);
    margin-bottom: 0.75rem;
}
.hpgr-kpi-label {
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.25rem;
}
.hpgr-kpi-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.2;
}
.hpgr-kpi-unit {
    font-size: 0.95rem;
    font-weight: 600;
    color: #CBD5E1;
    margin-left: 0.25rem;
}
.hpgr-kpi-sub {
    font-size: 0.78rem;
    color: #CBD5E1;
    margin-top: 0.3rem;
}
.hpgr-callout {
    border-radius: 12px;
    padding: 0.95rem 1.15rem;
    margin-bottom: 1rem;
    border: 1px solid transparent;
    box-shadow: 0 6px 16px rgba(2, 6, 23, 0.35);
}
.hpgr-callout-title {
    font-weight: 800;
    font-size: 0.98rem;
    margin-bottom: 0.28rem;
}
.hpgr-callout-body {
    font-size: 0.88rem;
    line-height: 1.48;
    color: #E2E8F0;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(19, 28, 49, 0.9);
    border-radius: 10px 10px 0 0;
    border: 1px solid rgba(148, 163, 184, 0.24);
    padding: 8px 16px;
    font-weight: 600;
    color: #CBD5E1;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.45) 0%, rgba(14, 165, 233, 0.45) 100%) !important;
    border-bottom: 3px solid #38BDF8 !important;
    color: #FFFFFF !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #0EA5E9 55%, #10B981 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.25) !important;
    font-weight: 800 !important;
    font-size: 1.02rem !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.35) !important;
}
button[kind="primary"]:hover {
    filter: brightness(1.1);
}
div[data-testid="stPageLink-NavLink"],
a[data-testid="stPageLink-NavLink"] {
    background: rgba(30, 41, 59, 0.78) !important;
    border: 1px solid rgba(56, 189, 248, 0.28) !important;
    border-radius: 10px !important;
    padding: 0.45rem 0.75rem !important;
}
</style>""",
        unsafe_allow_html=True,
    )


def render_hero_banner(
    title: str,
    subtitle: str,
    badges: Sequence[str] | None = None,
    gradient: str = "indigo-teal",
) -> None:
    """Render a colorful gradient hero banner with optional pill badges."""
    bg = GRADIENTS.get(gradient, GRADIENTS["indigo-teal"])
    badge_html = ""
    if badges:
        pills = "".join(f'<span class="hpgr-pill">{escape(str(b))}</span>' for b in badges)
        badge_html = f'<div class="hpgr-badge-row">{pills}</div>'

    html = (
        f'<div class="hpgr-hero" style="background:{bg};">'
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(subtitle)}</p>"
        f"{badge_html}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    unit: str = "",
    subtitle: str = "",
    accent: str = "#38BDF8",
    delta_text: str = "",
    delta_color: str = "#2DD4BF",
) -> None:
    """Render a colorful dark-glass KPI card with zero leading indentation (no Markdown code blocks)."""
    unit_html = f'<span class="hpgr-kpi-unit">{escape(unit)}</span>' if unit else ""
    delta_html = (
        f'<div style="font-size:0.82rem;font-weight:700;color:{delta_color};margin-top:0.2rem;">{escape(delta_text)}</div>'
        if delta_text
        else ""
    )
    sub_html = f'<div class="hpgr-kpi-sub">{escape(subtitle)}</div>' if subtitle else ""

    html = (
        f'<div class="hpgr-kpi" style="border-left:5px solid {accent};">'
        f'<div class="hpgr-kpi-label">{escape(label)}</div>'
        f'<div class="hpgr-kpi-val">{escape(value)}{unit_html}</div>'
        f"{delta_html}{sub_html}</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_confidence_banner(range_check: Dict[str, Any]) -> None:
    """Render a color-coded Emerald / Amber / Rose extrapolation warning banner."""
    level = range_check.get("confidence_level", "MODERATE_IN_RANGE")
    headline = range_check.get("headline", "")
    summary = range_check.get("summary", "")
    messages = range_check.get("messages", [])

    if level == "LOW_EXTRAPOLATION":
        bg, border, title_col = "rgba(159, 18, 57, 0.28)", "#FB7185", "#FDA4AF"
        icon = "🚨"
    elif level == "CAUTION_NEAR_BOUNDARY":
        bg, border, title_col = "rgba(180, 83, 9, 0.28)", "#FBBF24", "#FDE68A"
        icon = "⚠️"
    else:
        bg, border, title_col = "rgba(6, 95, 70, 0.28)", "#34D399", "#6EE7B7"
        icon = "✅"

    details_html = ""
    if messages:
        items = "".join(f"<li>{escape(m)}</li>" for m in messages)
        details_html = f'<ul style="margin:0.4rem 0 0 1.2rem;padding:0;color:#F1F5F9;">{items}</ul>'

    html = (
        f'<div class="hpgr-callout" style="background:{bg};border-color:{border};border-left:6px solid {border};">'
        f'<div class="hpgr-callout-title" style="color:{title_col};">{icon} {escape(headline)}</div>'
        f'<div class="hpgr-callout-body">{escape(summary)}{details_html}</div>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_bwi_unavailable_card() -> None:
    """Display an explicit notice that Bond Work Index / micro-cracking is not fabricated."""
    html = (
        '<div class="hpgr-callout" style="background:rgba(91, 33, 182, 0.26);border-color:#A78BFA;border-left:6px solid #A78BFA;">'
        '<div class="hpgr-callout-title" style="color:#DDD6FE;">'
        "🔬 Downstream Micro-Cracking / Bond Work Index (BWi) — Future Work (Not Available)"
        "</div>"
        '<div class="hpgr-callout-body">'
        "Before/after HPGR Bond Ball Mill Work Index measurements are <b>not present</b> in "
        "<code>HPGR_Extracted_Data.xlsx</code>. In accordance with strict scientific data integrity, "
        "no BWi reduction values are fabricated here. Supply paired feed vs. HPGR product Bond grindability "
        "test results to enable downstream micro-cracking modeling in a future release."
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_data_provenance_sidebar() -> None:
    """Render persistent provenance and limitation reminders in the Streamlit sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧪 Dataset Provenance")
        st.markdown(
            "- **Source:** UBC Thesis (Drozdiak, 2011)\n"
            "- **Ore:** Pilot-scale **Mesaba Cu-Ni sulphide**\n"
            "- **Records:** **21** real pilot tests (T1A01–T2B05)\n"
            "- **Press:** D = 0.750 m, L = 0.220 m\n"
            "- **Scope:** Offline research prototype only"
        )
        st.warning(
            "Not iron-ore plant data. Not validated for Extract Met equipment or live PLC/SCADA control."
        )

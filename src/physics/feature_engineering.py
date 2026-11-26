"""Physics-derived diagnostics and strictly separated synthetic exploration data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data_loader import PROJECT_ROOT
from src.physics.hpgr_equations import (
    DEFAULT_ROLL_DIAMETER_M,
    DEFAULT_ROLL_WIDTH_M,
    compute_force_balance_gap_mm,
    compute_micro_cracking_bwi,
    compute_net_specific_energy,
    compute_pressing_force_from_fsp,
    compute_reduction_ratio,
    compute_specific_pressing_force,
    compute_specific_throughput_constant,
    compute_throughput_from_mdot,
)


def add_physics_diagnostic_columns(
    df: pd.DataFrame,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
    width_m: float = DEFAULT_ROLL_WIDTH_M,
) -> pd.DataFrame:
    """Append non-destructive physics diagnostic columns for EDA and verification."""
    out = df.copy()
    out["calc_Fsp_N_mm2"] = out["pressing_force_set_kN"].apply(
        lambda f: round(compute_specific_pressing_force(f, diameter_m, width_m), 3)
        if pd.notna(f)
        else np.nan
    )
    out["calc_m_dot"] = out.apply(
        lambda r: round(
            compute_specific_throughput_constant(
                r["throughput_t_h"], r["roll_speed_set_m_s"], diameter_m, width_m
            ),
            2,
        )
        if pd.notna(r["throughput_t_h"]) and pd.notna(r["roll_speed_set_m_s"]) and r["roll_speed_set_m_s"] > 0
        else np.nan,
        axis=1,
    )
    out["calc_net_Esp_kWh_t"] = out.apply(
        lambda r: round(
            compute_net_specific_energy(r["power_draw_kW"], r["idle_power_kW"], r["throughput_t_h"]),
            3,
        )
        if pd.notna(r["power_draw_kW"]) and pd.notna(r["idle_power_kW"]) and pd.notna(r["throughput_t_h"])
        else np.nan,
        axis=1,
    )
    for col_src, col_dst in [
        ("scaleup_P80_mm", "reduction_ratio_scaleup"),
        ("centre_P80_mm", "reduction_ratio_centre"),
        ("edge_P80_mm", "reduction_ratio_edge"),
    ]:
        if col_src in out.columns:
            out[col_dst] = out.apply(
                lambda r: round(compute_reduction_ratio(r["feed_F80_mm"], r[col_src]), 2)
                if pd.notna(r["feed_F80_mm"]) and pd.notna(r[col_src]) and r[col_src] > 0
                else np.nan,
                axis=1,
            )
    return out


def generate_synthetic_exploration_data(
    config: Dict[str, Any],
    n_samples: int | None = None,
    profile: str | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate physically bounded synthetic rows solely for software testing/exploration.

    Supports both UBC Mesaba pilot bounds and PDF §5.3 literature ranges (345-419 t/hm3,
    product P80 0.15-0.80 mm, specific energy 0.9-2.2 kWh/t) with 2,000 points by default.
    Stored separately in data/synthetic/ with data_source = 'SYNTHETIC'.
    Never mixed into real training or evaluation datasets.
    """
    syn_cfg = config.get("synthetic_policy", {})
    geom = config.get("machine_geometry", {})
    gap_cfg = config.get("synthetic_gap_and_psd", {})
    bwi_cfg = config.get("micro_cracking_literature", {})

    d = float(geom.get("roll_diameter_m", DEFAULT_ROLL_DIAMETER_M))
    w = float(geom.get("roll_width_m", DEFAULT_ROLL_WIDTH_M))

    n_pts = int(n_samples if n_samples is not None else syn_cfg.get("n_samples", 2000))
    rng_seed = int(seed if seed is not None else syn_cfg.get("seed", 42))
    prof_key = profile or syn_cfg.get("default_profile", "ubc_mesaba_pilot")
    profiles = syn_cfg.get("profiles", {})
    prof_bounds = profiles.get(prof_key, syn_cfg.get("bounds", {}))

    rng = np.random.default_rng(rng_seed)

    fsp = rng.uniform(*prof_bounds.get("actual_specific_pressure", [2.0, 5.0]), size=n_pts)
    speed = rng.uniform(*prof_bounds.get("roll_speed_set_m_s", [0.60, 0.90]), size=n_pts)
    moisture = rng.uniform(*prof_bounds.get("feed_moisture_pct", [1.0, 5.8]), size=n_pts)
    f80 = rng.uniform(*prof_bounds.get("feed_F80_mm", [2.0, 22.0]), size=n_pts)

    base_mdot = float(prof_bounds.get("base_mdot", prof_bounds.get("assumed_specific_throughput_constant", 298.0)))
    mdot_lo, mdot_hi = prof_bounds.get("m_dot_range", [220.0, 330.0])
    m_dot_est = base_mdot - 4.0 * (fsp - 3.9) - 8.0 * (moisture - 2.5) + rng.normal(0.0, 8.0, n_pts)
    m_dot_est = np.clip(m_dot_est, float(mdot_lo), float(mdot_hi))

    throughput = np.array([compute_throughput_from_mdot(m, v, d, w) for m, v in zip(m_dot_est, speed)])
    pressing_force_kn = np.array([compute_pressing_force_from_fsp(p, d, w) for p in fsp])

    esp_lo, esp_hi = prof_bounds.get("esp_range", [0.75, 2.25])
    if prof_key == "pdf_iron_ore_ranges":
        net_esp = 0.28 * fsp + 0.07 * moisture + 0.015 * f80 + 0.05 + rng.normal(0.0, 0.06, n_pts)
        p80_lo, p80_hi = prof_bounds.get("p80_range", [0.15, 0.80])
        centre_p80 = np.clip(0.55 - 0.07 * (fsp - 3.5) + 0.012 * (f80 - 12.0) + rng.normal(0.0, 0.04, n_pts), float(p80_lo), float(p80_hi) * 0.9)
        edge_p80 = np.clip(centre_p80 * 1.45 + rng.normal(0.0, 0.03, n_pts), float(p80_lo), float(p80_hi))
    else:
        net_esp = 0.32 * fsp + 0.08 * moisture + 0.012 * f80 - 0.15 + rng.normal(0.0, 0.05, n_pts)
        p80_lo, p80_hi = prof_bounds.get("p80_range", [1.0, 12.0])
        centre_p80 = np.clip(0.32 * f80 - 0.65 * (fsp - 3.5) + 0.25 * (moisture - 2.5) + rng.normal(0.0, 0.35, n_pts), float(p80_lo), float(p80_hi) * 0.85)
        edge_p80 = np.clip(centre_p80 * 1.48 + rng.normal(0.0, 0.45, n_pts), float(p80_lo), float(p80_hi))

    net_esp = np.clip(net_esp, float(esp_lo), float(esp_hi))

    w_c = float(gap_cfg.get("centre_weight", 0.85))
    w_e = float(gap_cfg.get("edge_weight", 0.15))
    blended_p80 = w_c * centre_p80 + w_e * edge_p80

    gaps = np.array(
        [
            compute_force_balance_gap_mm(
                p,
                m,
                reference_gap_mm=float(gap_cfg.get("reference_gap_mm", 22.5)),
                reference_fsp_n_mm2=float(gap_cfg.get("reference_fsp_n_mm2", 3.95)),
                stiffness_slope_mm_per_fsp=float(gap_cfg.get("stiffness_slope_mm_per_fsp", -1.45)),
                moisture_slope_mm_per_pct=float(gap_cfg.get("moisture_slope_mm_per_pct", -0.95)),
                min_gap_mm=float(gap_cfg.get("min_gap_mm", 12.0)),
                max_gap_mm=float(gap_cfg.get("max_gap_mm", 30.0)),
            )
            + rng.normal(0.0, 0.35)
            for p, m in zip(fsp, moisture)
        ]
    )

    feed_bwi = float(bwi_cfg.get("default_feed_bwi_kwh_t", 15.0))
    bwi_slope = float(bwi_cfg.get("linear_reduction_pct_per_kwh_t", 6.0))
    bwi_cap = float(bwi_cfg.get("max_reduction_pct", 25.0))
    bwi_red_pct = np.array(
        [compute_micro_cracking_bwi(feed_bwi, e, bwi_slope, bwi_cap)["reduction_pct"] for e in net_esp]
    )

    df_syn = pd.DataFrame(
        {
            "Test_ID": [f"SYN_{i + 1:04d}" for i in range(n_pts)],
            "Phase": "Synthetic Exploration",
            "data_source": "SYNTHETIC",
            "synthetic_profile": prof_key,
            "actual_specific_pressure": np.round(fsp, 2),
            "pressing_force_set_kN": np.round(pressing_force_kn, 1),
            "roll_speed_set_m_s": np.round(speed, 2),
            "feed_moisture_pct": np.round(moisture, 2),
            "feed_F80_mm": np.round(f80, 2),
            "synthetic_force_balance_gap_mm": np.round(gaps, 2),
            "specific_throughput_constant": np.round(m_dot_est, 2),
            "throughput_t_h": np.round(throughput, 2),
            "net_specific_energy_kWh_t": np.round(net_esp, 3),
            "centre_P80_mm": np.round(centre_p80, 3),
            "edge_P80_mm": np.round(edge_p80, 3),
            "blended_85_15_P80_mm": np.round(blended_p80, 3),
            "simulated_bwi_reduction_pct": np.round(bwi_red_pct, 2),
        }
    )

    out_path = output_path or (PROJECT_ROOT / syn_cfg.get("output_file", "data/synthetic/hpgr_synthetic_exploration.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(out_path, index=False, encoding="utf-8")
    return df_syn

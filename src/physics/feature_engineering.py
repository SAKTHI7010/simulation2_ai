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
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate physically bounded synthetic rows solely for software testing/exploration.

    Stored separately in data/synthetic/ with data_source = 'SYNTHETIC'.
    Never mixed into real training or evaluation datasets.
    """
    syn_cfg = config.get("synthetic_policy", {})
    geom = config.get("machine_geometry", {})
    d = float(geom.get("roll_diameter_m", DEFAULT_ROLL_DIAMETER_M))
    w = float(geom.get("roll_width_m", DEFAULT_ROLL_WIDTH_M))

    n_samples = int(syn_cfg.get("n_samples", 40))
    seed = int(syn_cfg.get("seed", 42))
    bounds = syn_cfg.get("bounds", {})
    rng = np.random.default_rng(seed)

    fsp = rng.uniform(*bounds.get("actual_specific_pressure", [2.0, 5.0]), size=n_samples)
    speed = rng.uniform(*bounds.get("roll_speed_set_m_s", [0.60, 0.90]), size=n_samples)
    moisture = rng.uniform(*bounds.get("feed_moisture_pct", [1.0, 5.8]), size=n_samples)
    f80 = rng.uniform(*bounds.get("feed_F80_mm", [2.0, 22.0]), size=n_samples)

    base_mdot = float(bounds.get("assumed_specific_throughput_constant", 298.0))
    # Approximate physics-guided trends purely for synthetic exploration
    m_dot_est = base_mdot - 4.0 * (fsp - 3.9) - 8.0 * (moisture - 2.5) + rng.normal(0.0, 8.0, n_samples)
    m_dot_est = np.clip(m_dot_est, 220.0, 330.0)
    throughput = np.array([compute_throughput_from_mdot(m, v, d, w) for m, v in zip(m_dot_est, speed)])
    pressing_force_kn = np.array([compute_pressing_force_from_fsp(p, d, w) for p in fsp])

    net_esp = np.clip(0.32 * fsp + 0.08 * moisture + 0.012 * f80 - 0.15 + rng.normal(0.0, 0.05, n_samples), 0.75, 2.2)

    df_syn = pd.DataFrame(
        {
            "Test_ID": [f"SYN_{i + 1:03d}" for i in range(n_samples)],
            "Phase": "Synthetic Exploration",
            "data_source": "SYNTHETIC",
            "actual_specific_pressure": np.round(fsp, 2),
            "pressing_force_set_kN": np.round(pressing_force_kn, 1),
            "roll_speed_set_m_s": np.round(speed, 2),
            "feed_moisture_pct": np.round(moisture, 2),
            "feed_F80_mm": np.round(f80, 2),
            "specific_throughput_constant": np.round(m_dot_est, 2),
            "throughput_t_h": np.round(throughput, 2),
            "net_specific_energy_kWh_t": np.round(net_esp, 3),
        }
    )

    out_path = output_path or (PROJECT_ROOT / syn_cfg.get("output_file", "data/synthetic/hpgr_synthetic_exploration.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_syn.to_csv(out_path, index=False, encoding="utf-8")
    return df_syn

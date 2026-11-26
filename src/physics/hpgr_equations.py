"""Documented HPGR physics equations from Drozdiak (UBC, 2011) Appendix C."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_ROLL_DIAMETER_M = 0.750
DEFAULT_ROLL_WIDTH_M = 0.220


def _validate_positive(name: str, value: float) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"{name} must be a positive finite number (got {value}).")
    return val


def compute_specific_pressing_force(
    force_kn: float,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
    width_m: float = DEFAULT_ROLL_WIDTH_M,
) -> float:
    """Compute specific pressing force Fsp = F / (D * L) in N/mm^2 (MN/m^2).

    Since force is in kN and D * L is in m^2, dividing kN/m^2 by 1000 yields MN/m^2 = N/mm^2.
    """
    d = _validate_positive("diameter_m", diameter_m)
    w = _validate_positive("width_m", width_m)
    f = float(force_kn)
    if f < 0.0:
        raise ValueError(f"force_kn must be non-negative (got {force_kn}).")
    return (f / 1000.0) / (d * w)


def compute_pressing_force_from_fsp(
    fsp_n_mm2: float,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
    width_m: float = DEFAULT_ROLL_WIDTH_M,
) -> float:
    """Compute total pressing force F (kN) = Fsp (N/mm^2) * D * L * 1000."""
    d = _validate_positive("diameter_m", diameter_m)
    w = _validate_positive("width_m", width_m)
    fsp = float(fsp_n_mm2)
    if fsp < 0.0:
        raise ValueError(f"fsp_n_mm2 must be non-negative (got {fsp_n_mm2}).")
    return fsp * d * w * 1000.0


def compute_peripheral_speed(
    rpm: float,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
) -> float:
    """Compute roll peripheral speed v = pi * D * rpm / 60 in m/s."""
    d = _validate_positive("diameter_m", diameter_m)
    r = float(rpm)
    if r < 0.0:
        raise ValueError(f"rpm must be non-negative (got {rpm}).")
    return math.pi * d * r / 60.0


def compute_rpm_from_speed(
    roll_speed_m_s: float,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
) -> float:
    """Compute roll rotational speed (rpm) = 60 * v / (pi * D)."""
    d = _validate_positive("diameter_m", diameter_m)
    v = float(roll_speed_m_s)
    if v < 0.0:
        raise ValueError(f"roll_speed_m_s must be non-negative (got {roll_speed_m_s}).")
    return 60.0 * v / (math.pi * d)


def compute_specific_throughput_constant(
    throughput_t_h: float,
    roll_speed_m_s: float,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
    width_m: float = DEFAULT_ROLL_WIDTH_M,
) -> float:
    """Compute specific throughput constant m-dot = W / (D * L * v) in t*s / (m^3*h)."""
    d = _validate_positive("diameter_m", diameter_m)
    w = _validate_positive("width_m", width_m)
    v = _validate_positive("roll_speed_m_s", roll_speed_m_s)
    wt = float(throughput_t_h)
    if wt < 0.0:
        raise ValueError(f"throughput_t_h must be non-negative (got {throughput_t_h}).")
    return wt / (d * w * v)


def compute_throughput_from_mdot(
    m_dot: float,
    roll_speed_m_s: float,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
    width_m: float = DEFAULT_ROLL_WIDTH_M,
) -> float:
    """Compute throughput W (t/h) = m-dot * D * L * v."""
    d = _validate_positive("diameter_m", diameter_m)
    w = _validate_positive("width_m", width_m)
    v = _validate_positive("roll_speed_m_s", roll_speed_m_s)
    md = float(m_dot)
    if md < 0.0:
        raise ValueError(f"m_dot must be non-negative (got {m_dot}).")
    return md * d * w * v


def compute_net_specific_energy(
    power_draw_kw: float,
    idle_power_kw: float,
    throughput_t_h: float,
) -> float:
    """Compute net specific energy Esp_net = (Pt - Pi) / W in kWh/t."""
    wt = _validate_positive("throughput_t_h", throughput_t_h)
    return (float(power_draw_kw) - float(idle_power_kw)) / wt


def compute_total_specific_energy(
    power_draw_kw: float,
    throughput_t_h: float,
) -> float:
    """Compute total specific energy Esp_total = Pt / W in kWh/t."""
    wt = _validate_positive("throughput_t_h", throughput_t_h)
    return float(power_draw_kw) / wt


def compute_reduction_ratio(feed_f80_mm: float, product_p80_mm: float) -> float:
    """Compute comminution reduction ratio F80 / P80."""
    p80 = _validate_positive("product_p80_mm", product_p80_mm)
    f80 = float(feed_f80_mm)
    if f80 <= 0.0:
        raise ValueError(f"feed_f80_mm must be positive (got {feed_f80_mm}).")
    return f80 / p80


def audit_physics_consistency(
    df: pd.DataFrame,
    diameter_m: float = DEFAULT_ROLL_DIAMETER_M,
    width_m: float = DEFAULT_ROLL_WIDTH_M,
) -> pd.DataFrame:
    """Verify source workbook values against documented HPGR physics equations."""
    rows = []
    for _, r in df.iterrows():
        calc_fsp = compute_specific_pressing_force(r["pressing_force_set_kN"], diameter_m, width_m)
        calc_mdot = compute_specific_throughput_constant(
            r["throughput_t_h"], r["roll_speed_set_m_s"], diameter_m, width_m
        )
        calc_net_esp = compute_net_specific_energy(
            r["power_draw_kW"], r["idle_power_kW"], r["throughput_t_h"]
        )
        calc_tot_esp = compute_total_specific_energy(r["power_draw_kW"], r["throughput_t_h"])

        obs_fsp = float(r["actual_specific_pressure"])
        obs_mdot = float(r["specific_throughput_constant"])
        obs_net = float(r["net_specific_energy_kWh_t"])
        obs_tot = float(r["total_specific_energy_kWh_t"])

        rows.append(
            {
                "Test_ID": r["Test_ID"],
                "Phase": r["Phase"],
                "obs_Fsp_N_mm2": obs_fsp,
                "calc_Fsp_N_mm2": round(calc_fsp, 3),
                "diff_Fsp": round(calc_fsp - obs_fsp, 4),
                "obs_m_dot": obs_mdot,
                "calc_m_dot": round(calc_mdot, 2),
                "diff_m_dot": round(calc_mdot - obs_mdot, 2),
                "obs_net_Esp_kWh_t": obs_net,
                "calc_net_Esp_kWh_t": round(calc_net_esp, 3),
                "diff_net_Esp": round(calc_net_esp - obs_net, 4),
                "obs_tot_Esp_kWh_t": obs_tot,
                "calc_tot_Esp_kWh_t": round(calc_tot_esp, 3),
                "diff_tot_Esp": round(calc_tot_esp - obs_tot, 4),
            }
        )
    return pd.DataFrame(rows)


def compute_micro_cracking_bwi(
    feed_bwi_kwh_t: float,
    net_specific_energy_kwh_t: float,
    reduction_pct_per_kwh_t: float = 6.0,
    max_reduction_pct: float = 25.0,
) -> Dict[str, float]:
    """Literature-calibrated linear micro-cracking BWi reduction model (PDF §5.3 Ex. 5).

    Not measured in HPGR_Extracted_Data.xlsx; clearly labeled as synthetic/literature estimate.
    """
    bwi_in = _validate_positive("feed_bwi_kwh_t", feed_bwi_kwh_t)
    esp = max(0.0, float(net_specific_energy_kwh_t))
    slope = max(0.0, float(reduction_pct_per_kwh_t))
    cap = max(0.0, min(50.0, float(max_reduction_pct)))
    reduction_pct = min(cap, slope * esp)
    product_bwi = bwi_in * (1.0 - reduction_pct / 100.0)
    return {
        "feed_bwi_kwh_t": round(bwi_in, 2),
        "product_bwi_kwh_t": round(product_bwi, 2),
        "bwi_delta_kwh_t": round(bwi_in - product_bwi, 2),
        "reduction_pct": round(reduction_pct, 2),
    }


def compute_force_balance_gap_mm(
    fsp_n_mm2: float,
    moisture_pct: float = 2.5,
    reference_gap_mm: float = 22.5,
    reference_fsp_n_mm2: float = 3.95,
    stiffness_slope_mm_per_fsp: float = -1.45,
    moisture_slope_mm_per_pct: float = -0.95,
    min_gap_mm: float = 12.0,
    max_gap_mm: float = 30.0,
) -> float:
    """Synthetic force-balance floating roll gap approximation (PDF §5.3 Ex. 1)."""
    fsp = max(0.0, float(fsp_n_mm2))
    moist = max(0.0, float(moisture_pct))
    gap = (
        float(reference_gap_mm)
        + float(stiffness_slope_mm_per_fsp) * (fsp - float(reference_fsp_n_mm2))
        + float(moisture_slope_mm_per_pct) * (moist - 2.5)
    )
    return float(max(float(min_gap_mm), min(float(max_gap_mm), gap)))


def compute_blended_product_psd(
    centre_p80_mm: float,
    edge_p80_mm: float,
    centre_weight: float = 0.85,
    edge_weight: float = 0.15,
    m_centre: float = 0.95,
    m_edge: float = 1.15,
    sieve_sizes_mm: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, float]:
    """Generate Rosin-Rammler centre/edge PSD curves and weighted blend (default 85:15)."""
    c_p80 = _validate_positive("centre_p80_mm", centre_p80_mm)
    e_p80 = _validate_positive("edge_p80_mm", edge_p80_mm)
    w_c = float(centre_weight)
    w_e = float(edge_weight)
    if w_c < 0.0 or w_e < 0.0 or (w_c + w_e) <= 0.0:
        raise ValueError("centre_weight and edge_weight must be non-negative with positive sum.")
    total_w = w_c + w_e
    w_c /= total_w
    w_e /= total_w

    sizes = (
        np.asarray(sieve_sizes_mm, dtype=float)
        if sieve_sizes_mm is not None
        else np.array([0.045, 0.075, 0.106, 0.15, 0.212, 0.30, 0.425, 0.60, 0.85, 1.18, 1.70, 2.36, 3.35, 4.75, 6.70, 9.50, 13.2, 19.0])
    )
    sizes = np.sort(sizes)
    k_80 = -math.log(0.20)  # ~1.6094379

    pass_c = 100.0 * (1.0 - np.exp(-k_80 * (sizes / c_p80) ** float(m_centre)))
    pass_e = 100.0 * (1.0 - np.exp(-k_80 * (sizes / e_p80) ** float(m_edge)))
    pass_blend = w_c * pass_c + w_e * pass_e

    psd_df = pd.DataFrame(
        {
            "size_mm": sizes,
            "centre_passing_pct": np.round(pass_c, 2),
            "edge_passing_pct": np.round(pass_e, 2),
            "blended_passing_pct": np.round(pass_blend, 2),
        }
    )

    # Interpolate blended P80 on log-size grid
    dense_sizes = np.geomspace(max(sizes.min(), 1e-3), max(sizes.max(), e_p80 * 2.5), 400)
    dense_c = 100.0 * (1.0 - np.exp(-k_80 * (dense_sizes / c_p80) ** float(m_centre)))
    dense_e = 100.0 * (1.0 - np.exp(-k_80 * (dense_sizes / e_p80) ** float(m_edge)))
    dense_blend = w_c * dense_c + w_e * dense_e
    blended_p80 = float(np.interp(80.0, dense_blend, dense_sizes))
    return psd_df, round(blended_p80, 3)

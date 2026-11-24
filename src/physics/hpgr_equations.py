"""Documented HPGR physics equations from Drozdiak (UBC, 2011) Appendix C."""

from __future__ import annotations

import math
from typing import Union

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

"""Unit tests for documented HPGR physics equations."""

from __future__ import annotations

import math

import pytest

from src.data_loader import load_operating_data
from src.physics.hpgr_equations import (
    audit_physics_consistency,
    compute_net_specific_energy,
    compute_peripheral_speed,
    compute_pressing_force_from_fsp,
    compute_reduction_ratio,
    compute_rpm_from_speed,
    compute_specific_pressing_force,
    compute_specific_throughput_constant,
    compute_throughput_from_mdot,
    compute_total_specific_energy,
)


def test_specific_pressing_force_roundtrip() -> None:
    # T1A01: F = 810 kN, D = 0.750 m, L = 0.220 m -> ~4.91 N/mm^2
    fsp = compute_specific_pressing_force(810.0, diameter_m=0.750, width_m=0.220)
    assert fsp == pytest.approx(4.9091, abs=1e-3)
    f_back = compute_pressing_force_from_fsp(fsp, diameter_m=0.750, width_m=0.220)
    assert f_back == pytest.approx(810.0, abs=1e-6)


def test_specific_throughput_constant_roundtrip() -> None:
    # T1A01: W = 35.98 t/h, v = 0.75 m/s, D = 0.750 m, L = 0.220 m -> ~290.75
    mdot = compute_specific_throughput_constant(35.98, 0.75, 0.750, 0.220)
    assert mdot == pytest.approx(290.747, abs=1e-2)
    w_back = compute_throughput_from_mdot(mdot, 0.75, 0.750, 0.220)
    assert w_back == pytest.approx(35.98, abs=1e-6)


def test_peripheral_speed_and_rpm() -> None:
    v = compute_peripheral_speed(19.1, diameter_m=0.750)
    assert v == pytest.approx(0.750, abs=1e-2)
    rpm = compute_rpm_from_speed(v, diameter_m=0.750)
    assert rpm == pytest.approx(19.1, abs=1e-6)


def test_net_and_total_specific_energy() -> None:
    # T1A01: Pt = 77.94 kW, Pi = 8.12 kW, W = 35.98 t/h
    net_esp = compute_net_specific_energy(77.94, 8.12, 35.98)
    tot_esp = compute_total_specific_energy(77.94, 35.98)
    assert net_esp == pytest.approx(1.9405, abs=1e-3)
    assert tot_esp == pytest.approx(2.1662, abs=1e-3)


def test_reduction_ratio() -> None:
    assert compute_reduction_ratio(21.33, 7.15) == pytest.approx(2.9832, abs=1e-3)


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        compute_specific_pressing_force(500.0, diameter_m=0.0, width_m=0.22)
    with pytest.raises(ValueError):
        compute_net_specific_energy(50.0, 8.0, throughput_t_h=0.0)
    with pytest.raises(ValueError):
        compute_reduction_ratio(20.0, product_p80_mm=-1.0)


def test_physics_audit_matches_workbook_closely() -> None:
    df = load_operating_data()
    audit = audit_physics_consistency(df, diameter_m=0.750, width_m=0.220)
    assert len(audit) == 21
    assert audit["diff_net_Esp"].abs().max() < 0.01
    assert audit["diff_tot_Esp"].abs().max() < 0.01
    assert audit["diff_Fsp"].abs().max() < 0.02

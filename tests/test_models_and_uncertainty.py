"""Integration tests for model training, prediction, uncertainty, and range checks."""

from __future__ import annotations

import pandas as pd

from src.models.predict import (
    generate_sensitivity_curve,
    load_trained_artifacts,
    predict_all_targets,
    predict_scenario_comparison,
)
from src.models.uncertainty import check_input_ranges


def test_trained_artifacts_present_and_all_targets_available() -> None:
    bundle = load_trained_artifacts(auto_train=True)
    meta = bundle["metadata"]
    assert set(meta["targets"].keys()) == {
        "throughput_t_h",
        "net_specific_energy_kWh_t",
        "scaleup_P80_mm",
        "centre_P80_mm",
        "edge_P80_mm",
    }
    assert meta["targets"]["throughput_t_h"]["n_samples"] == 21
    assert meta["targets"]["scaleup_P80_mm"]["n_samples"] == 16
    assert len(bundle["eval_df"]) == 15  # 5 targets x 3 models


def test_in_range_and_out_of_range_detection() -> None:
    bundle = load_trained_artifacts(auto_train=True)
    bounds = bundle["metadata"]["global_feature_bounds"]

    in_pt = {
        "actual_specific_pressure": 3.95,
        "roll_speed_set_m_s": 0.75,
        "feed_moisture_pct": 2.5,
        "feed_F80_mm": 15.0,
    }
    chk_in = check_input_ranges(in_pt, bounds)
    assert chk_in["is_extrapolation"] is False
    assert chk_in["confidence_level"] == "MODERATE_IN_RANGE"

    out_pt = {**in_pt, "actual_specific_pressure": 6.2}
    chk_out = check_input_ranges(out_pt, bounds)
    assert chk_out["is_extrapolation"] is True
    assert chk_out["confidence_level"] == "LOW_EXTRAPOLATION"
    assert "actual_specific_pressure" in chk_out["out_of_range_features"]


def test_prediction_and_scenario_outputs() -> None:
    bundle = load_trained_artifacts(auto_train=True)
    base = {
        "actual_specific_pressure": 3.95,
        "roll_speed_set_m_s": 0.75,
        "feed_moisture_pct": 2.5,
        "feed_F80_mm": 21.33,
    }
    mod = {**base, "actual_specific_pressure": 4.50}

    pred = predict_all_targets(base, model_key="gpr", artifacts=bundle)
    assert pred["predictions"]["throughput_t_h"]["gpr_std"] > 0.0
    assert pred["predictions"]["throughput_t_h"]["ci95_high"] > pred["predictions"]["throughput_t_h"]["ci95_low"]

    scen = predict_scenario_comparison(base, mod, model_key="gpr", artifacts=bundle)
    assert len(scen["comparison_df"]) == 5

    sweep = generate_sensitivity_curve(base, "actual_specific_pressure", [2.5, 4.0, 6.0], "net_specific_energy_kWh_t", artifacts=bundle)
    assert list(sweep["in_training_range"]) == [True, True, False]

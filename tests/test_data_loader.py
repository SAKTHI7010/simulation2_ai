"""Unit tests for data loading, validation, and non-fabrication rules."""

from __future__ import annotations

import pytest

from src.data_loader import (
    WorkbookLoadError,
    build_data_dictionary,
    load_operating_data,
    load_physics_config,
)
from src.data_validation import DataValidationError, validate_operating_data
from src.preprocessing import get_target_modeling_frame


def test_load_real_operating_data_preserves_21_rows_and_ids() -> None:
    df = load_operating_data()
    assert len(df) == 21
    assert df["Test_ID"].nunique() == 21
    assert "T1A01" in df["Test_ID"].values
    assert "T2B05" in df["Test_ID"].values
    assert (df["data_source"] == "REAL").all()


def test_missing_workbook_raises_helpful_error() -> None:
    with pytest.raises(WorkbookLoadError, match="does not exist"):
        load_operating_data("non_existent_workbook_12345.xlsx")


def test_validation_reports_exact_usable_rows_without_fabricating_p80() -> None:
    config = load_physics_config()
    df = load_operating_data()
    features = [f["name"] for f in config["selected_features"]]
    targets = [t["name"] for t in config["targets"]]

    report = validate_operating_data(df, features, targets, raise_on_error=True)
    assert report.is_valid is True
    assert report.target_availability["throughput_t_h"]["usable_rows"] == 21
    assert report.target_availability["net_specific_energy_kWh_t"]["usable_rows"] == 21
    assert report.target_availability["scaleup_P80_mm"]["usable_rows"] == 16
    assert report.target_availability["centre_P80_mm"]["usable_rows"] == 16
    assert report.target_availability["edge_P80_mm"]["usable_rows"] == 16
    assert "static_gap_mm" in report.constant_columns


def test_validation_raises_when_required_column_missing() -> None:
    config = load_physics_config()
    df = load_operating_data().drop(columns=["feed_F80_mm"])
    features = [f["name"] for f in config["selected_features"]]
    targets = [t["name"] for t in config["targets"]]

    with pytest.raises(DataValidationError, match="feed_F80_mm"):
        validate_operating_data(df, features, targets, raise_on_error=True)


def test_get_target_modeling_frame_drops_missing_targets() -> None:
    config = load_physics_config()
    df = load_operating_data()
    features = [f["name"] for f in config["selected_features"]]
    X, y, meta = get_target_modeling_frame(df, features, "centre_P80_mm")
    assert len(X) == 16
    assert len(y) == 16
    assert y.notna().all()
    assert "T2A01" not in meta["Test_ID"].values


def test_data_dictionary_contains_all_columns() -> None:
    df = load_operating_data()
    dd = build_data_dictionary(df)
    assert len(dd) == len(df.columns)
    assert {"column", "unit", "missing", "unique_count"}.issubset(dd.columns)

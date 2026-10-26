import math

from evaluation.input_semantics import (
    FindingKind,
    PrimaryState,
    RainfallProvenance,
    build_prediction_input,
    record_rainfall_clamp,
    summarize_rainfall_response,
)
from evaluation.validation import FEATURES
import pandas as pd


def valid_values():
    return {
        "N": 40, "P": 30, "K": 50, "temperature": 25,
        "humidity": 70, "ph": 6.5, "rainfall": 100,
    }


def test_feature_order_and_numeric_fields_are_preserved():
    result = build_prediction_input(valid_values())
    assert [field.feature for field in result.fields] == FEATURES
    assert list(result.values) == FEATURES
    assert result.primary_state is PrimaryState.VALIDATED


def test_missing_malformed_nan_and_infinity_are_invalid():
    values = valid_values()
    values.pop("rainfall")
    values["N"] = "not-a-number"
    values["P"] = float("nan")
    values["K"] = math.inf
    result = build_prediction_input(values)
    assert result.primary_state is PrimaryState.INVALID
    assert sum(item.kind is FindingKind.INVALID_INPUT for item in result.findings) >= 4


def test_categorical_proxy_metadata_is_retained_without_claiming_conversion():
    metadata = {
        "N": {
            "origin": "categorical_level_proxy",
            "original_value": "Medium",
            "transformations": ("training_data_median_proxy",),
            "unit_status": "unverified",
        }
    }
    result = build_prediction_input(valid_values(), field_metadata=metadata)
    field = result.fields[0]
    assert field.origin == "categorical_level_proxy"
    assert field.original_value == "Medium"
    assert result.primary_state is PrimaryState.SEMANTICS_UNRESOLVED
    assert any(item.kind is FindingKind.UNRESOLVED_SEMANTICS for item in result.findings)


def test_ocr_metadata_with_and_without_uncertainty():
    certain = build_prediction_input(valid_values(), field_metadata={
        "N": {"origin": "ocr", "original_value": 120, "ocr_confidence": 0.91}
    })
    assert not any(item.kind is FindingKind.OCR_UNCERTAINTY for item in certain.findings)

    uncertain = build_prediction_input(valid_values(), field_metadata={
        "N": {"origin": "ocr", "original_value": 120, "ocr_uncertain": True}
    })
    assert any(item.kind is FindingKind.OCR_UNCERTAINTY for item in uncertain.findings)
    assert uncertain.primary_state is PrimaryState.SEMANTICS_UNRESOLVED


def test_rainfall_provenance_records_transformation_and_compatibility():
    rainfall = RainfallProvenance(
        source="Open-Meteo archive",
        original_value=300.0,
        original_unit="mm",
        start_date="2026-01-01",
        end_date="2026-03-31",
        interval="90 days",
        aggregation_method="sum daily precipitation then divide by 3",
        transformed_value=100.0,
        transformations=("divide_by_3",),
        training_feature_compatibility=False,
    )
    result = build_prediction_input(valid_values(), rainfall=rainfall)
    assert result.rainfall == rainfall
    assert result.primary_state is PrimaryState.SEMANTICS_UNRESOLVED
    assert any(item.feature == "rainfall" for item in result.findings)


def test_unknown_units_missing_metadata_and_out_of_range_are_multiple_findings():
    values = valid_values()
    values["N"] = 999
    result = build_prediction_input(
        values,
        training_ranges={feature: {"min": 0, "q01": 1, "q99": 100, "max": 100} for feature in FEATURES},
        field_metadata={"N": {
            "unit_status": "unknown",
            "source_metadata_required": True,
        }},
    )
    kinds = {item.kind for item in result.findings}
    assert FindingKind.OUTSIDE_TRAINING_RANGE in kinds
    assert FindingKind.UNRESOLVED_SEMANTICS in kinds
    assert FindingKind.MISSING_SOURCE_METADATA in kinds
    assert result.primary_state is PrimaryState.SEMANTICS_UNRESOLVED


def test_primary_state_precedence():
    values = valid_values()
    values["N"] = -1
    result = build_prediction_input(values, field_metadata={
        "P": {"unit_status": "unknown"},
    })
    assert result.primary_state is PrimaryState.INVALID


def test_serialization_round_trip():
    result = build_prediction_input(valid_values(), field_metadata={
        "rainfall": {"origin": "weather_api", "source_metadata": {"provider": "Open-Meteo"}}
    })
    restored = type(result).from_dict(result.to_dict())
    assert restored == result


def test_rainfall_response_summary_preserves_arithmetic_and_metadata():
    daily = pd.DataFrame({
        "time": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "precipitation_sum": [10.0, 20.0, 30.0],
    })
    result = summarize_rainfall_response(
        daily,
        source="mock Open-Meteo",
        requested_start_date="2026-01-01",
        requested_end_date="2026-03-31",
        divisor=3,
        original_unit="mm",
        source_metadata={"endpoint": "mock"},
    )
    assert result.original_value == 60.0
    assert result.transformed_value == 20.0
    assert result.original_unit == "mm"
    assert result.actual_start_date == "2026-01-01"
    assert result.actual_end_date == "2026-01-03"
    assert result.observation_count == 3
    assert result.missing_precipitation_count == 0
    assert result.training_feature_compatibility is None


def test_rainfall_summary_matches_existing_missing_value_sum_behavior():
    daily = pd.DataFrame({
        "time": ["2026-01-01", "2026-01-03"],
        "precipitation_sum": [10.0, float("nan")],
    })
    result = summarize_rainfall_response(
        daily,
        source=None,
        requested_start_date="2026-01-01",
        requested_end_date="2026-01-03",
        divisor=3,
    )
    assert result.original_value == 10.0
    assert result.transformed_value == 10.0 / 3
    assert result.missing_precipitation_count == 1
    assert result.observation_count == 2


def test_rainfall_clamping_is_recorded_without_changing_final_value():
    daily = pd.DataFrame({
        "time": ["2026-01-01"],
        "precipitation_sum": [1200.0],
    })
    rainfall = summarize_rainfall_response(
        daily,
        source="mock",
        requested_start_date="2026-01-01",
        requested_end_date="2026-03-31",
        divisor=3,
    )
    clamped = record_rainfall_clamp(rainfall, final_value=300.0, lower=20.0, upper=300.0)
    assert clamped.transformed_value == 300.0
    assert clamped.clamping_history == ("clamped to [20, 300]",)

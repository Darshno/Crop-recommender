import pandas as pd

from evaluation.validation import (
    FEATURES,
    TARGET,
    build_training_ranges,
    validate_dataset,
    validate_input_values,
)


def sample_frame():
    rows = []
    for label, offset in (("rice", 0.0), ("maize", 1.0)):
        for i in range(3):
            rows.append({
                "N": 20 + i + offset,
                "P": 30 + i,
                "K": 40 + i,
                "temperature": 22 + i,
                "humidity": 60 + i,
                "ph": 6 + i * 0.1,
                "rainfall": 100 + i * 5,
                "label": label,
            })
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])


def test_valid_dataset_passes():
    report = validate_dataset(sample_frame())
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["class_count"] == 2


def test_missing_required_column_fails():
    frame = sample_frame().drop(columns=["ph"])
    report = validate_dataset(frame)
    assert report["valid"] is False
    assert "ph" in report["missing_columns"]


def test_missing_value_fails():
    frame = sample_frame()
    frame.loc[0, "N"] = float("nan")
    report = validate_dataset(frame)
    assert report["valid"] is False
    assert "missing values" in " ".join(report["errors"]).lower()


def test_non_numeric_feature_fails():
    frame = sample_frame()
    # Newer pandas versions reject assigning a string directly into an int64 column.
    # Make the column object dtype to model malformed/mixed-type input data.
    frame["temperature"] = frame["temperature"].astype("object")
    frame.loc[0, "temperature"] = "hot"
    report = validate_dataset(frame)
    assert report["valid"] is False
    assert "temperature" in report["non_numeric_values_by_feature"]


def test_physically_implausible_input_is_rejected():
    checked = validate_input_values({
        "N": -1, "P": 20, "K": 20, "temperature": 25,
        "humidity": 60, "ph": 6.5, "rainfall": 100,
    })
    assert checked["valid"] is False
    assert any("N=" in msg for msg in checked["errors"])


def test_missing_prediction_field_is_rejected():
    values = {key: 1 for key in FEATURES if key != "rainfall"}
    checked = validate_input_values(values)
    assert checked["valid"] is False
    assert any("rainfall" in msg for msg in checked["errors"])


def test_prediction_input_warns_about_training_extrapolation():
    ranges = {f: {"min": 1, "q01": 2, "q99": 9, "max": 10} for f in FEATURES}
    values = {f: 5 for f in FEATURES}
    values["N"] = 12
    checked = validate_input_values(values, ranges)
    assert checked["valid"] is True
    assert any("extrapolation" in msg for msg in checked["warnings"])


def test_training_ranges_include_distribution_summary():
    frame = sample_frame()
    ranges = build_training_ranges(frame[FEATURES])
    assert set(ranges) == set(FEATURES)
    for feature in FEATURES:
        assert ranges[feature]["min"] <= ranges[feature]["q01"]
        assert ranges[feature]["q01"] <= ranges[feature]["median"]
        assert ranges[feature]["median"] <= ranges[feature]["q99"]
        assert ranges[feature]["q99"] <= ranges[feature]["max"]

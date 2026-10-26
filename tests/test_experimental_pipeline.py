import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.independent_data_v1.experimental_pipeline import (
    FEATURES,
    REQUIRED,
    normalize_columns,
    normalize_label,
    prepare_frame,
)


def test_schema_normalization_and_labels():
    frame = pd.DataFrame([[1, 2, 3, 20, 50, 6, 100, " Rice "]],
                         columns=[" n ", "P", "K", "Temperature", "humidity", "ph", "rainfall", "Crop"])
    normalized, mapping = normalize_columns(frame)
    assert set(REQUIRED).issubset(normalized.columns)
    assert mapping[" n "] == "N"
    assert normalize_label("  RICE  ") == "rice"


def test_prepare_frame_reports_invalid_values(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([[1, 2, 3, "bad", 50, 6, 100, "rice"]],
                 columns=REQUIRED).to_csv(path, index=False)
    with pytest.raises(ValueError, match="failed safe validation"):
        prepare_frame(path, "test")


def test_prepare_frame_reports_missing_values(tmp_path):
    path = tmp_path / "missing.csv"
    row = [1, 2, 3, 20, np.nan, 6, 100, "rice"]
    pd.DataFrame([row], columns=REQUIRED).to_csv(path, index=False)
    with pytest.raises(ValueError, match="failed safe validation"):
        prepare_frame(path, "test")


def test_source_traceability_and_duplicate_inputs(tmp_path):
    path = tmp_path / "data.csv"
    row = [1, 2, 3, 20, 50, 6, 100, "rice"]
    pd.DataFrame([row, row], columns=REQUIRED).to_csv(path, index=False)
    frame, report = prepare_frame(path, "independent")
    assert report["exact_duplicate_full_rows"] == 1
    assert frame["source"].tolist() == ["independent", "independent"]
    assert frame["source_row"].tolist() == [2, 3]


def test_nonfinite_values_are_rejected(tmp_path):
    path = tmp_path / "inf.csv"
    row = [1, 2, 3, np.inf, 50, 6, 100, "rice"]
    pd.DataFrame([row], columns=REQUIRED).to_csv(path, index=False)
    with pytest.raises(ValueError, match="failed safe validation"):
        prepare_frame(path, "test")

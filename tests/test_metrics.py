import numpy as np
import pytest

from evaluation.metrics import (
    calibration_bin_table,
    classification_metrics,
    expected_calibration_error,
    multiclass_brier_score,
)


def test_perfect_probabilities_have_zero_ece_and_brier():
    y = np.array([0, 1, 2, 0])
    p = np.eye(3)[y]
    assert expected_calibration_error(y, p) == pytest.approx(0.0)
    assert multiclass_brier_score(y, p) == pytest.approx(0.0)


def test_calibration_table_counts_all_samples():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.7, 0.3]])
    bins = calibration_bin_table(y, p, n_bins=5)
    assert len(bins) == 5
    assert int(bins["count"].sum()) == len(y)


def test_classification_metrics_return_expected_keys():
    y = np.array([0, 1, 2, 0])
    p = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
    ])
    result = classification_metrics(y, p, labels=[0, 1, 2], top_k=3)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["top_3_accuracy"] == pytest.approx(1.0)
    assert "top_label_ece_10_bins" in result
    assert "multiclass_brier_score" in result


def test_invalid_probability_rows_are_rejected():
    y = np.array([0, 1])
    p = np.array([[0.8, 0.3], [0.2, 0.8]])
    with pytest.raises(ValueError, match="sum to 1"):
        expected_calibration_error(y, p)


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        classification_metrics([0, 1], [[0.8, 0.2], [0.1, 0.9]], labels=[0, 1], top_k=0)

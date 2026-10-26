import numpy as np
import pandas as pd
import pytest

from evaluation.dataset_audit import audit_dataset
from evaluation.validation import FEATURES, TARGET


def sample_frame() -> pd.DataFrame:
    rows = []
    for label_id, label in enumerate(["rice", "maize", "cotton"]):
        for i in range(6):
            rows.append({
                "N": 10 + label_id * 20 + i,
                "P": 20 + label_id * 10 + i,
                "K": 30 + label_id * 5 + i,
                "temperature": 18.0 + label_id * 3 + i * 0.11,
                "humidity": 45.0 + label_id * 8 + i * 0.13,
                "ph": 5.0 + label_id * 0.5 + i * 0.017,
                "rainfall": 50.0 + label_id * 20 + i * 0.19,
                "label": label,
            })
    return pd.DataFrame(rows, columns=FEATURES + [TARGET])


def test_clean_dataset_has_no_duplicates_or_split_overlap():
    report, tables = audit_dataset(sample_frame(), test_size=0.25, random_state=7)
    assert report["integrity"]["exact_duplicate_full_rows_beyond_first"] == 0
    assert report["integrity"]["exact_duplicate_feature_rows_beyond_first"] == 0
    assert report["integrity"]["exact_conflicting_feature_group_count"] == 0
    assert report["train_test_leakage"]["exact_feature_vector_overlap_count"] == 0
    assert report["train_test_leakage"]["exact_full_row_overlap_count"] == 0
    assert len(tables["nearest_train_neighbors"]) == 5


def test_identical_features_with_different_labels_are_detected():
    frame = sample_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate[TARGET] = "maize"
    frame = pd.concat([frame, duplicate], ignore_index=True)
    report, _ = audit_dataset(frame, test_size=0.25, random_state=7)
    assert report["integrity"]["exact_duplicate_feature_rows_beyond_first"] == 1
    assert report["integrity"]["exact_conflicting_feature_group_count"] == 1
    assert report["integrity"]["rows_in_exact_conflicting_feature_groups"] == 2


def test_exact_duplicate_full_rows_are_detected():
    frame = sample_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report, _ = audit_dataset(frame, test_size=0.25, random_state=7)
    assert report["integrity"]["exact_duplicate_full_rows_beyond_first"] == 1


def test_label_order_blocks_are_reported():
    frame = sample_frame()
    report, _ = audit_dataset(frame, test_size=0.25, random_state=7)
    assert report["row_order"]["each_label_one_contiguous_block"] is True
    assert report["row_order"]["unshuffled_tail_holdout_diagnostic_only"]["classes_in_tail_test"] < 3


def test_non_numeric_and_missing_data_do_not_silently_pass():
    frame = sample_frame()
    frame["temperature"] = frame["temperature"].astype(object)
    frame.loc[0, "temperature"] = "hot"
    with pytest.raises(ValueError, match="numeric"):
        audit_dataset(frame)

    frame = sample_frame()
    frame.loc[0, "temperature"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        audit_dataset(frame)

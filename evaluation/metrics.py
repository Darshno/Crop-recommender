"""Classification metrics and probability-calibration diagnostics."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)


def _check_probabilities(y_true: Any, probabilities: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] != len(y):
        raise ValueError("probabilities must be a 2D array with one row per target")
    if p.shape[1] < 2:
        raise ValueError("probabilities must contain at least two class columns")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain non-finite values")
    if (p < -1e-9).any() or (p > 1 + 1e-9).any():
        raise ValueError("probabilities must lie between 0 and 1")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each probability row must sum to 1")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("target labels must be integer class indices matching probability columns")
    return y, p


def calibration_bin_table(y_true: Any, probabilities: Any, n_bins: int = 10) -> pd.DataFrame:
    """Build equal-width top-label reliability bins, retaining empty bins."""
    y, p = _check_probabilities(y_true, probabilities)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    confidence = p.max(axis=1)
    predicted = p.argmax(axis=1)
    correct = (predicted == y).astype(float)
    bin_ids = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        rows.append({
            "bin": int(bin_id),
            "lower_bound": float(bin_id / n_bins),
            "upper_bound": float((bin_id + 1) / n_bins),
            "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()) if mask.any() else np.nan,
            "accuracy": float(correct[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: Any, probabilities: Any, n_bins: int = 10) -> float:
    """Top-label expected calibration error (lower is better)."""
    y, p = _check_probabilities(y_true, probabilities)
    table = calibration_bin_table(y, p, n_bins=n_bins)
    non_empty = table[table["count"] > 0]
    confidence_accuracy_gap = (non_empty["accuracy"] - non_empty["mean_confidence"]).abs()
    weights = non_empty["count"] / max(len(y), 1)
    return float((weights * confidence_accuracy_gap).sum())


def multiclass_brier_score(y_true: Any, probabilities: Any, n_classes: int | None = None) -> float:
    """Mean summed squared error against one-hot labels; lower is better."""
    y, p = _check_probabilities(y_true, probabilities)
    actual_classes = p.shape[1]
    if n_classes is not None and actual_classes != n_classes:
        raise ValueError("n_classes does not match probability columns")
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def classification_metrics(
    y_true: Any,
    probabilities: Any,
    labels: Sequence[int],
    top_k: int = 3,
) -> dict[str, Any]:
    """Return classification and calibration metrics for encoded class labels."""
    y, p = _check_probabilities(y_true, probabilities)
    label_array = np.asarray(labels, dtype=int)
    if len(label_array) != p.shape[1]:
        raise ValueError("labels must contain one entry per probability column")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    predicted = label_array[p.argmax(axis=1)]
    effective_k = min(top_k, len(label_array))
    return {
        "sample_count": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_precision": float(precision_score(y, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "top_3_accuracy": float(top_k_accuracy_score(y, p, k=effective_k, labels=label_array)),
        "log_loss": float(log_loss(y, p, labels=label_array)),
        "multiclass_brier_score": multiclass_brier_score(y, p, n_classes=len(label_array)),
        "top_label_ece_10_bins": expected_calibration_error(y, p, n_bins=10),
    }

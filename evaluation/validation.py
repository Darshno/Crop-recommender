"""Dataset and prediction-input validation for the crop recommender.

The checks are intentionally conservative: they catch malformed data and
physically implausible inputs, but do not pretend to be an agronomist.
"""
from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# Broad physical plausibility bounds, not crop-suitability thresholds.
HARD_BOUNDS = {
    "N": (0.0, 10000.0),
    "P": (0.0, 10000.0),
    "K": (0.0, 10000.0),
    "temperature": (-20.0, 65.0),
    "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0),
    "rainfall": (0.0, 10000.0),
}


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a reproducible diagnostics report; do not silently repair data."""
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    report: dict[str, Any] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "required_features": FEATURES,
        "target": TARGET,
        "missing_columns": missing_columns,
        "missing_values_by_column": {},
        "non_numeric_values_by_feature": {},
        "non_finite_values_by_feature": {},
        "domain_violations_by_feature": {},
        "blank_labels": 0,
        "duplicate_full_rows": 0,
        "conflicting_duplicate_feature_rows": 0,
        "class_count": 0,
        "class_counts": {},
        "warnings": [],
        "errors": [],
        "valid": False,
    }

    if missing_columns:
        report["errors"].append(f"Missing required columns: {', '.join(missing_columns)}")
        return report

    required = df[FEATURES + [TARGET]].copy()
    report["missing_values_by_column"] = {
        str(k): int(v) for k, v in required.isna().sum().items() if int(v) > 0
    }
    if report["missing_values_by_column"]:
        report["errors"].append("Required columns contain missing values.")

    numeric_frame = pd.DataFrame(index=df.index)
    for feature in FEATURES:
        numeric = pd.to_numeric(df[feature], errors="coerce")
        bad = df[feature].notna() & numeric.isna()
        if int(bad.sum()):
            report["non_numeric_values_by_feature"][feature] = int(bad.sum())
        numeric_frame[feature] = numeric
        finite_mask = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        non_finite_count = int((numeric.notna().to_numpy() & ~finite_mask).sum())
        if non_finite_count:
            report["non_finite_values_by_feature"][feature] = non_finite_count
        lower, upper = HARD_BOUNDS[feature]
        violation = numeric.notna() & ((numeric < lower) | (numeric > upper))
        if int(violation.sum()):
            report["domain_violations_by_feature"][feature] = int(violation.sum())

    labels = df[TARGET].astype("string").str.strip()
    report["blank_labels"] = int((labels.isna() | labels.eq("")).sum())
    nonblank = labels.dropna()
    nonblank = nonblank[nonblank.ne("")]
    report["class_count"] = int(nonblank.nunique())
    report["class_counts"] = {
        str(k): int(v) for k, v in nonblank.value_counts().sort_index().items()
    }

    report["duplicate_full_rows"] = int(required.duplicated().sum())
    if report["duplicate_full_rows"]:
        report["warnings"].append(
            f"Found {report['duplicate_full_rows']} exact duplicate rows; inspect before training."
        )

    # Same measurements paired with conflicting targets can cause a model to
    # appear to learn contradictory examples. Compare exact numeric vectors.
    safe = numeric_frame.copy()
    safe[TARGET] = labels
    valid_rows = safe[FEATURES].notna().all(axis=1) & safe[TARGET].notna() & safe[TARGET].ne("")
    duplicate_features = safe.loc[valid_rows].groupby(FEATURES, dropna=False)[TARGET].nunique()
    conflicts = int((duplicate_features > 1).sum())
    report["conflicting_duplicate_feature_rows"] = conflicts
    if conflicts:
        report["errors"].append(
            f"Found {conflicts} identical feature vectors associated with different crop labels."
        )

    if report["non_numeric_values_by_feature"]:
        report["errors"].append("One or more features contain non-numeric values.")
    if report["non_finite_values_by_feature"]:
        report["errors"].append("One or more features contain infinite values.")
    if report["domain_violations_by_feature"]:
        report["errors"].append("One or more feature values fall outside broad physical plausibility bounds.")
    if report["blank_labels"]:
        report["errors"].append("Target labels contain blank values.")
    if report["class_count"] < 2:
        report["errors"].append("At least two crop classes are required.")

    # A warning rather than a hard failure: exact duplicates may be legitimate,
    # but they can inflate random-split evaluation if left unexplained.
    report["valid"] = not report["errors"]
    return report


def validate_input_values(
    values: Mapping[str, Any],
    training_ranges: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Validate one prediction request and flag extrapolation beyond training data."""
    errors: list[str] = []
    warnings: list[str] = []
    cleaned: dict[str, float] = {}

    for feature in FEATURES:
        if feature not in values:
            errors.append(f"Missing input: {feature}")
            continue
        try:
            value = float(values[feature])
        except (TypeError, ValueError):
            errors.append(f"{feature} must be numeric.")
            continue
        if not np.isfinite(value):
            errors.append(f"{feature} must be finite.")
            continue
        low, high = HARD_BOUNDS[feature]
        if value < low or value > high:
            errors.append(f"{feature}={value:g} is outside the permitted input range [{low:g}, {high:g}].")
            continue
        cleaned[feature] = value

    if not errors and training_ranges:
        for feature in FEATURES:
            bounds = training_ranges.get(feature, {})
            value = cleaned[feature]
            low, high = bounds.get("min"), bounds.get("max")
            q01, q99 = bounds.get("q01"), bounds.get("q99")
            if low is not None and value < low:
                warnings.append(
                    f"{feature}={value:g} is below the training minimum ({low:g}); this is an extrapolation."
                )
            elif high is not None and value > high:
                warnings.append(
                    f"{feature}={value:g} is above the training maximum ({high:g}); this is an extrapolation."
                )
            elif q01 is not None and value < q01:
                warnings.append(
                    f"{feature}={value:g} is below the training 1st percentile ({q01:g}); it is unusual in the training data."
                )
            elif q99 is not None and value > q99:
                warnings.append(
                    f"{feature}={value:g} is above the training 99th percentile ({q99:g}); it is unusual in the training data."
                )

    return {"valid": not errors, "errors": errors, "warnings": warnings, "values": cleaned}


def build_training_ranges(x: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Store min/max and central quantiles for each feature in the training split."""
    ranges: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        s = pd.to_numeric(x[feature], errors="raise")
        ranges[feature] = {
            "min": float(s.min()),
            "q01": float(s.quantile(0.01)),
            "median": float(s.median()),
            "q99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return ranges

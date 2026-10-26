"""Safe, reproducible experiment combining the original and independent data.

Run from the project root:
    .venv\\Scripts\\python.exe experiments\\independent_data_v1\\experimental_pipeline.py

The default source for the original dataset is Crop_recommendation.csv because
that is the original file present in this repository. The script never writes
to either source CSV or to crop_model.pkl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
REQUIRED = FEATURES + [TARGET]
RANDOM_STATE = 42
HARD_BOUNDS = {
    "N": (0.0, 10000.0), "P": (0.0, 10000.0), "K": (0.0, 10000.0),
    "temperature": (-20.0, 65.0), "humidity": (0.0, 100.0),
    "ph": (0.0, 14.0), "rainfall": (0.0, 10000.0),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().casefold().replace("_", " ").split())


def normalize_label(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    aliases = {normalize_name(name): name for name in REQUIRED}
    aliases.update({"crop": TARGET, "crop label": TARGET, "crop name": TARGET, "target": TARGET})
    normalized: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for original in frame.columns:
        key = normalize_name(original)
        if key in aliases:
            canonical = aliases[key]
            if canonical in normalized:
                raise ValueError(f"Ambiguous columns map to {canonical!r}: {normalized[canonical]!r}, {original!r}")
            normalized[canonical] = original
            mapping[str(original)] = canonical
    missing = [column for column in REQUIRED if column not in normalized]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {', '.join(missing)}")
    result = frame.rename(columns={source: target for source, target in mapping.items()})
    return result, mapping


def prepare_frame(path: Path, source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"{source} dataset does not exist: {path}")
    raw = pd.read_csv(path)
    frame, column_mapping = normalize_columns(raw)
    working = frame.loc[:, REQUIRED].copy()
    original_labels = working[TARGET].astype("string")
    working["original_label"] = original_labels
    working[TARGET] = original_labels.map(normalize_label)
    working["source"] = source
    working["source_row"] = np.arange(len(working), dtype=int) + 2
    missing = {str(k): int(v) for k, v in working[REQUIRED].isna().sum().items() if int(v)}
    non_numeric: dict[str, int] = {}
    non_finite: dict[str, int] = {}
    ranges: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        converted = pd.to_numeric(working[feature], errors="coerce")
        non_numeric[feature] = int((working[feature].notna() & converted.isna()).sum())
        finite = np.isfinite(converted.to_numpy(dtype=float, na_value=np.nan))
        non_finite[feature] = int((converted.notna().to_numpy() & ~finite).sum())
        working[feature] = converted
        if converted.notna().any():
            ranges[feature] = {"min": float(converted.min()), "max": float(converted.max())}
    blank_labels = int(working[TARGET].isna().sum() + working[TARGET].eq("").sum())
    domain_violations = {
        feature: int(((working[feature] < low) | (working[feature] > high)).sum())
        for feature, (low, high) in HARD_BOUNDS.items()
    }
    report = {
        "path": str(path.resolve()), "source": source, "row_count": int(len(raw)),
        "column_count": int(len(raw.columns)), "original_columns": [str(c) for c in raw.columns],
        "normalized_columns": REQUIRED, "column_mapping": column_mapping,
        "dtypes": {str(k): str(v) for k, v in raw.dtypes.items()},
        "missing_values": missing, "non_numeric_values": {k: v for k, v in non_numeric.items() if v},
        "non_finite_values": {k: v for k, v in non_finite.items() if v},
        "blank_labels": blank_labels, "domain_violations": {k: v for k, v in domain_violations.items() if v},
        "ranges": ranges, "unique_labels": sorted(working[TARGET].dropna().unique().tolist()),
        "label_mapping": {
            str(label): sorted(working.loc[working[TARGET].eq(label), "original_label"].dropna().unique().tolist())
            for label in sorted(working[TARGET].dropna().unique().tolist())
        },
        "class_distribution": {str(k): int(v) for k, v in working[TARGET].value_counts().sort_index().items()},
        "exact_duplicate_full_rows": int(working[REQUIRED].duplicated().sum()),
    }
    if missing or any(non_numeric.values()) or any(non_finite.values()) or blank_labels or any(domain_violations.values()):
        raise ValueError(f"{source} failed safe validation; see data_audit.json for diagnostics.")
    return working, report


def feature_key(row: pd.Series) -> tuple[Any, ...]:
    return tuple(row[FEATURES].tolist())


def full_key(row: pd.Series) -> tuple[Any, ...]:
    return feature_key(row) + (row[TARGET],)


def metric_tables(y_true: pd.Series, predictions: np.ndarray, labels: list[str]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    report = classification_report(
        y_true, predictions, labels=labels, output_dict=True, zero_division=0
    )
    present = sorted(set(y_true))
    metrics = {
        "sample_count": int(len(y_true)),
        "class_count_in_evaluation": int(len(present)),
        "class_coverage": int(len(present)),
        "classes_present": present,
        "classes_absent": [label for label in labels if label not in present],
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }
    per_class = pd.DataFrame([
        {"class": label, "support": int(report.get(label, {}).get("support", 0)),
         "precision": float(report.get(label, {}).get("precision", 0.0)),
         "recall": float(report.get(label, {}).get("recall", 0.0)),
         "f1": float(report.get(label, {}).get("f1-score", 0.0))}
        for label in labels
    ])
    matrix = pd.DataFrame(confusion_matrix(y_true, predictions, labels=labels), index=labels, columns=labels)
    matrix.index.name = "actual"
    matrix.columns.name = "predicted"
    return metrics, per_class, matrix


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original_path = args.original.resolve()
    independent_path = args.independent.resolve()
    baseline_path = args.baseline.resolve()
    protected_before = {str(path): sha256(path) for path in (original_path, independent_path, baseline_path)}
    original, original_report = prepare_frame(original_path, "original")
    independent, independent_report = prepare_frame(independent_path, "independent")

    all_rows = pd.concat([original, independent], ignore_index=True)
    original_full_keys = set(map(tuple, original[REQUIRED].itertuples(index=False, name=None)))
    independent_full_keys = set(map(tuple, independent[REQUIRED].itertuples(index=False, name=None)))
    exact_counts = {
        "within_original": int(original[REQUIRED].duplicated().sum()),
        "within_independent": int(independent[REQUIRED].duplicated().sum()),
        "across_files": int(len(original_full_keys.intersection(independent_full_keys))),
    }
    grouped = all_rows.groupby(FEATURES, dropna=False, sort=False)
    conflict_groups = grouped[TARGET].nunique()
    conflict_keys = set(conflict_groups[conflict_groups > 1].index.tolist())
    conflict_rows = all_rows[all_rows.set_index(FEATURES).index.isin(conflict_keys)].copy()
    conflict_rows.to_csv(output / "conflicting_labels.csv", index=False)

    all_rows["_full_key"] = all_rows.apply(full_key, axis=1)
    deduped = all_rows.drop_duplicates("_full_key", keep="first").drop(columns="_full_key")
    deduped["feature_key"] = deduped.apply(feature_key, axis=1)
    deduped["conflict"] = deduped["feature_key"].isin(conflict_keys)
    safe_rows = deduped[~deduped["conflict"]].copy()

    original_feature_keys = set(original.apply(feature_key, axis=1))
    independent_candidates = independent[
        ~independent.apply(feature_key, axis=1).isin(original_feature_keys)
    ].copy()
    independent_candidates["feature_key"] = independent_candidates.apply(feature_key, axis=1)
    independent_candidates = independent_candidates.drop_duplicates("feature_key", keep="first")
    independent_candidates = independent_candidates[~independent_candidates["feature_key"].isin(conflict_keys)]
    baseline_bundle = joblib.load(baseline_path)
    if not isinstance(baseline_bundle, dict) or not {"model", "encoder", "features"} <= set(baseline_bundle):
        raise ValueError("Baseline crop_model.pkl does not have the expected model/encoder/features bundle.")
    if list(baseline_bundle["features"]) != FEATURES:
        raise ValueError("Baseline feature order is incompatible with the expected seven features.")
    baseline_classes = [str(x) for x in baseline_bundle["encoder"].classes_]
    eligible = independent_candidates[independent_candidates[TARGET].isin(baseline_classes)].copy()
    incompatible = int(len(independent_candidates) - len(eligible))
    heldout_keys = set(eligible["feature_key"])
    train_rows = safe_rows[~safe_rows["feature_key"].isin(heldout_keys)].copy()
    train_rows.drop(columns=["feature_key", "conflict"]).loc[:, REQUIRED + ["original_label", "source", "source_row"]].to_csv(
        output / "cleaned_combined_dataset.csv", index=False
    )
    eligible.drop(columns=["feature_key"]).to_csv(output / "eligible_evaluation.csv", index=False)

    training_labels = sorted(train_rows[TARGET].unique().tolist())
    encoder = LabelEncoder().fit(training_labels)
    model = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=1
    )
    model.fit(train_rows[FEATURES].astype(float), encoder.transform(train_rows[TARGET]))
    experimental_bundle = {
        "model": model, "encoder": encoder, "features": FEATURES,
        "metadata": {"experiment": "independent_data_v1", "random_state": RANDOM_STATE}
    }
    model_path = output / "crop_model_experimental_v1.pkl"
    encoder_path = output / "label_encoder_experimental_v1.pkl"
    joblib.dump(experimental_bundle, model_path)
    joblib.dump(encoder, encoder_path)

    predictions_exp = model.predict(eligible[FEATURES].astype(float))
    exp_labels = encoder.inverse_transform(predictions_exp)
    experimental_metrics, exp_per_class, exp_matrix = metric_tables(eligible[TARGET], exp_labels, training_labels)
    baseline_model = baseline_bundle["model"]
    baseline_predictions = baseline_bundle["encoder"].inverse_transform(
        baseline_model.predict(eligible[FEATURES].astype(float))
    )
    baseline_metrics, baseline_per_class, baseline_matrix = metric_tables(eligible[TARGET], baseline_predictions, baseline_classes)
    exp_per_class.insert(0, "model", "experimental")
    baseline_per_class.insert(0, "model", "baseline")
    pd.concat([baseline_per_class, exp_per_class], ignore_index=True).to_csv(
        output / "per_class_metrics.csv", index=False
    )
    exp_matrix.to_csv(output / "confusion_matrix.csv")
    baseline_matrix.to_csv(output / "baseline_confusion_matrix.csv")

    audit = {
        "schema": {"features": FEATURES, "target": TARGET, "normalization": "trim and casefold names and labels"},
        "sources": {"original": original_report, "independent": independent_report},
        "unit_compatibility": {
            "status": "not_verified",
            "reason": "CSV files contain no unit/provenance metadata; values were not converted or clipped.",
            "feature_range_comparison": {
                feature: {"original": original_report["ranges"].get(feature), "independent": independent_report["ranges"].get(feature)}
                for feature in FEATURES
            },
        },
        "protected_input_sha256": protected_before,
    }
    write_json(output / "data_audit.json", audit)
    distribution = pd.concat([
        original.assign(dataset="original"), independent.assign(dataset="independent"),
        train_rows.assign(dataset="experimental_training"), eligible.assign(dataset="eligible_evaluation"),
    ]).groupby(["dataset", TARGET]).size().rename("count").reset_index()
    distribution.to_csv(output / "class_distribution.csv", index=False)
    write_json(output / "duplicate_summary.json", {
        **exact_counts, "conflicting_feature_groups": int(len(conflict_keys)),
        "conflicting_rows": int(len(conflict_rows)), "deduplicated_rows": int(len(deduped)),
        "training_rows": int(len(train_rows)), "heldout_rows": int(len(eligible)),
        "removed_original_duplicates_from_evaluation": int(len(independent) - len(independent_candidates)),
        "incompatible_label_rows_excluded_from_comparison": incompatible,
        "conflict_rows_excluded_from_training": int(len(deduped) - len(safe_rows)),
    })
    write_json(output / "preprocessing_summary.json", {
        "feature_order": FEATURES, "target": TARGET, "source_column_retained": True,
        "source_column_excluded_from_features": True, "label_normalization": "strip/casefold/whitespace collapse",
        "exact_duplicate_definition": "normalized seven-feature vector plus normalized label",
        "conflicts_excluded": True, "heldout_feature_vectors_excluded_from_training": True,
        "random_state": RANDOM_STATE, "library_versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
    })
    write_json(output / "training_report.json", {
        "status": "trained", "model_path": str(model_path), "encoder_path": str(encoder_path),
        "training_rows": int(len(train_rows)), "training_source_counts": {str(k): int(v) for k, v in train_rows["source"].value_counts().items()},
        "classes": training_labels, "evaluation_protocol": "independent rows not present in original or experimental training; baseline-compatible labels only",
        "eligible_evaluation_rows": int(len(eligible)), "experimental": experimental_metrics,
        "baseline": baseline_metrics, "baseline_classes": baseline_classes,
        "per_class_metrics": {
            "experimental": exp_per_class.to_dict(orient="records"),
            "baseline": baseline_per_class.to_dict(orient="records"),
        },
        "evidence_sufficient_for_generalization_claim": False,
        "limitations": ["Units and provenance are not verified.", "Independent labels absent from baseline encoder are excluded from the comparison.", "A generalization claim requires domain verification and sufficient representative coverage."],
        "protected_input_sha256_after": {str(path): sha256(path) for path in (original_path, independent_path, baseline_path)},
    })
    write_json(output / "metadata.json", experimental_bundle["metadata"] | {"features": FEATURES, "classes": training_labels})
    for path, before in protected_before.items():
        if sha256(Path(path)) != before:
            raise RuntimeError(f"Protected input changed unexpectedly: {path}")
    return {"output": str(output), "training_rows": len(train_rows), "eligible_rows": len(eligible), "conflicts": len(conflict_keys), "metrics": {"baseline": baseline_metrics, "experimental": experimental_metrics}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=PROJECT_ROOT / "Crop_recommendation.csv")
    parser.add_argument("--independent", type=Path, default=PROJECT_ROOT / "independent_data.csv")
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "crop_model.pkl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "independent_data_v1")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

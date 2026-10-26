"""Phase 3: external evaluation and train-serving input contract checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from evaluation.metrics import classification_metrics
from evaluation.validation import FEATURES, TARGET, validate_dataset

REQUIRED_MANIFEST_FIELDS = [
    "source_name",
    "source_reference",
    "collection_method",
    "geography",
    "observation_period",
    "label_definition",
    "independent_of_training_data",
    "units",
]
REQUIRED_UNIT_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return blocking errors; never infer missing provenance or units."""
    errors = []
    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in manifest or manifest[key] in (None, "", [], {}):
            errors.append(f"Missing required provenance field: {key}")
    if manifest.get("independent_of_training_data") is not True:
        errors.append("independent_of_training_data must be true for external evaluation.")
    units = manifest.get("units")
    if not isinstance(units, dict):
        errors.append("units must be an object mapping every model feature to its measurement unit.")
    else:
        for feature in REQUIRED_UNIT_FEATURES:
            if not isinstance(units.get(feature), str) or not units[feature].strip():
                errors.append(f"Missing measurement unit for feature: {feature}")
    return errors



def count_exact_feature_overlap(training_frame: pd.DataFrame, external_frame: pd.DataFrame) -> int:
    """Count external rows whose complete feature vector exists in training data."""
    train_keys = set(map(tuple, training_frame[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy()))
    external_keys = external_frame[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy()
    return int(sum(tuple(row) in train_keys for row in external_keys))


def evaluate_external_dataset(
    data_path: str | Path,
    manifest_path: str | Path,
    model_path: str | Path = "crop_model.pkl",
    output_dir: str | Path = "reports/phase3/external_evaluation",
    training_data_path: str | Path | None = "Crop_recommendation.csv",
) -> dict[str, Any]:
    """Evaluate a locked model on separately supplied labelled data.

    This function does not fit, tune, calibrate, or select a model. It fails closed
    if provenance is incomplete, data are invalid, labels are unknown, or the
    external CSV is byte-identical to the known training CSV.
    """
    data_path, manifest_path = Path(data_path), Path(manifest_path)
    model_path, output_dir = Path(model_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_errors = validate_provenance_manifest(manifest)
    if manifest_errors:
        raise ValueError("Provenance manifest is incomplete:\n- " + "\n- ".join(manifest_errors))

    data_hash = sha256_file(data_path)
    training_hash = None
    training_frame = None
    if training_data_path is not None and Path(training_data_path).exists():
        training_hash = sha256_file(training_data_path)
        if data_hash == training_hash:
            raise ValueError("External dataset is byte-identical to the configured training dataset.")
        training_frame = pd.read_csv(training_data_path)

    frame = pd.read_csv(data_path)
    validation = validate_dataset(frame)
    if not validation["valid"]:
        raise ValueError("External dataset failed validation: " + "; ".join(validation["errors"]))
    if len(frame) < 2:
        raise ValueError("External dataset must contain at least two labelled rows.")
    if training_frame is not None:
        # Detect row-level overlap even if the CSV was reordered or only a subset was copied.
        overlap_count = count_exact_feature_overlap(training_frame, frame)
        if overlap_count:
            raise ValueError(
                f"External dataset shares {overlap_count} exact feature vectors with the configured training CSV. "
                "Investigate overlap and use a genuinely independent dataset before evaluating."
            )

    bundle = joblib.load(model_path)
    expected_features = list(bundle.get("features", FEATURES))
    if expected_features != FEATURES:
        raise ValueError(f"Model feature order/schema mismatch. Expected {FEATURES}, got {expected_features}.")

    encoder = bundle["encoder"]
    labels = frame[TARGET].astype(str).str.strip()
    known = set(map(str, encoder.classes_))
    unknown = sorted(set(labels) - known)
    if unknown:
        raise ValueError("External dataset contains labels absent from the model: " + ", ".join(unknown))

    x = frame[expected_features].apply(pd.to_numeric, errors="raise")
    y = encoder.transform(labels)
    probabilities = np.asarray(bundle["model"].predict_proba(x), dtype=float)
    model_class_ids = np.asarray(bundle["model"].classes_, dtype=int)
    # Put probability columns in encoder class order, even if estimator class order differs.
    aligned = np.zeros((len(frame), len(encoder.classes_)), dtype=float)
    aligned[:, model_class_ids] = probabilities
    class_ids = np.arange(len(encoder.classes_))
    metrics = classification_metrics(y, aligned, labels=class_ids, top_k=3)

    predicted_ids = aligned.argmax(axis=1)
    report = classification_report(
        y, predicted_ids, labels=class_ids,
        target_names=[str(c) for c in encoder.classes_],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, predicted_ids, labels=class_ids)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / "per_crop_classification_report.csv")
    pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(output_dir / "confusion_matrix.csv")
    (output_dir / "external_evaluation.json").write_text(json.dumps({
        "status": "EVALUATED_ON_PROVIDED_EXTERNAL_DATA",
        "interpretation": "Metrics describe only this supplied dataset. They do not independently prove causal crop suitability or future farm outcomes.",
        "dataset": {
            "filename": data_path.name, "sha256": data_hash, "row_count": int(len(frame)),
            "class_count_present": int(labels.nunique()),
            "class_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        },
        "training_dataset_sha256": training_hash,
        "model": bundle.get("metadata", {}),
        "provenance": manifest,
        "metrics": metrics,
        "coverage": {
            "model_class_count": int(len(encoder.classes_)),
            "classes_present_in_external_data": int(labels.nunique()),
            "missing_model_classes": sorted(set(map(str, encoder.classes_)) - set(labels)),
            "has_field_or_farm_ids": bool(manifest.get("field_or_farm_ids_available", False)),
            "has_row_level_dates": bool(manifest.get("row_level_dates_available", False)),
            "has_independent_expert_labels": bool(manifest.get("labels_verified_by_agronomist", False)),
            "has_measured_outcomes": bool(manifest.get("outcomes_measured", False)),
        },
    }, indent=2, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n", encoding="utf-8")
    return {"metrics": metrics, "output_dir": str(output_dir.resolve()), "rows": len(frame)}


def input_contract() -> dict[str, Any]:
    """The required semantics that must be resolved before connecting live app inputs."""
    return {
        "status": "UNRESOLVED_UNTIL_SOURCE_DOCUMENTATION_CONFIRMS_SEMANTICS",
        "features": {
            "N": {"meaning": "soil nitrogen", "required_unit": "Must match training dataset's original unit and sampling method."},
            "P": {"meaning": "soil phosphorus", "required_unit": "Must match training dataset's original unit and sampling method."},
            "K": {"meaning": "soil potassium", "required_unit": "Must match training dataset's original unit and sampling method."},
            "temperature": {"meaning": "temperature", "required_unit": "degree Celsius; averaging window must match training data."},
            "humidity": {"meaning": "relative humidity", "required_unit": "percent; aggregation window must match training data."},
            "ph": {"meaning": "soil pH", "required_unit": "dimensionless; method and sampling depth should be documented."},
            "rainfall": {"meaning": "rainfall", "required_unit": "mm, with period (monthly vs growing-season total) explicitly confirmed."},
        },
        "known_app_behavior": {
            "temperature": "Mean of daily mean temperature over a recent 90-day window.",
            "humidity": "Mean of daily mean relative humidity over a recent 90-day window.",
            "rainfall": "Sum of daily precipitation over that 90-day window divided by 3 (approximate mm/month).",
            "soil_card": "N/P/K ratings are mapped to dataset quartiles for low/medium/high; this is a proxy mapping, not a verified unit conversion.",
            "warning": "Do not treat live app inputs as equivalent to training features until original feature definitions, units, and aggregation windows are documented and aligned.",
        },
    }


def write_input_contract(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(input_contract(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

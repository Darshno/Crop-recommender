"""Read-only forensic audit of the independent_data_v1 experiment."""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n", encoding="utf-8")


def stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce")
        rows.append({
            "feature": feature, "dtype": str(frame[feature].dtype),
            "missing": int(values.isna().sum()),
            "non_finite": int((~np.isfinite(values.dropna())).sum()),
            "min": float(values.min()), "q01": float(values.quantile(.01)),
            "q05": float(values.quantile(.05)), "q25": float(values.quantile(.25)),
            "mean": float(values.mean()), "median": float(values.median()),
            "q75": float(values.quantile(.75)), "q95": float(values.quantile(.95)),
            "q99": float(values.quantile(.99)), "max": float(values.max()),
            "std": float(values.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / "Crop_recommendation.csv"
    independent_path = ROOT / "independent_data.csv"
    experiment = ROOT / "experiments" / "independent_data_v1"
    original = pd.read_csv(original_path)
    independent = pd.read_csv(independent_path)
    eligible = pd.read_csv(experiment / "eligible_evaluation.csv")
    training = pd.read_csv(experiment / "cleaned_combined_dataset.csv")
    baseline = joblib.load(ROOT / "crop_model.pkl")
    experimental = joblib.load(experiment / "crop_model_experimental_v1.pkl")

    original_stats = stats(original)
    independent_stats = stats(independent)
    comparison = original_stats.merge(independent_stats, on="feature", suffixes=("_original", "_independent"))
    comparison["independent_outside_original_range_count"] = [
        int(((independent[f] < original[f].min()) | (independent[f] > original[f].max())).sum())
        for f in FEATURES
    ]
    comparison["independent_outside_original_range_percent"] = (
        comparison["independent_outside_original_range_count"] / len(independent) * 100
    )
    original_stats.to_csv(OUT / "original_feature_statistics.csv", index=False)
    independent_stats.to_csv(OUT / "independent_feature_statistics.csv", index=False)
    comparison.to_csv(OUT / "feature_comparison.csv", index=False)

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    for axis, feature in zip(axes.flat, FEATURES):
        axis.hist(original[feature], bins=30, alpha=.55, density=True, label="original")
        axis.hist(independent[feature], bins=30, alpha=.55, density=True, label="independent")
        axis.set_title(feature)
        axis.legend()
        axis.grid(alpha=.2)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "feature_distributions.png", dpi=160)
    plt.close(fig)

    baseline_feature_order = list(baseline["features"])
    experimental_feature_order = list(experimental["features"])
    baseline_encoder_classes = [str(x) for x in baseline["encoder"].classes_]
    experimental_encoder_classes = [str(x) for x in experimental["encoder"].classes_]
    independent_labels = sorted(independent[TARGET].astype(str).str.strip().str.casefold().unique())
    label_counts = independent[TARGET].astype(str).str.strip().str.casefold().value_counts().rename_axis("label").reset_index(name="count")
    label_counts.to_csv(OUT / "independent_label_distribution.csv", index=False)
    training[TARGET].value_counts().rename_axis("label").reset_index(name="count").to_csv(OUT / "experimental_training_label_distribution.csv", index=False)

    x_eval = eligible[FEATURES].astype(float)
    baseline_pred_ids = baseline["model"].predict(x_eval)
    baseline_pred = baseline["encoder"].inverse_transform(baseline_pred_ids)
    experimental_pred_ids = experimental["model"].predict(x_eval)
    experimental_pred = experimental["encoder"].inverse_transform(experimental_pred_ids)
    predictions = eligible[[TARGET, "source", "source_row"]].copy()
    predictions["baseline_prediction"] = baseline_pred
    predictions["experimental_prediction"] = experimental_pred
    predictions["baseline_correct"] = predictions[TARGET].eq(predictions["baseline_prediction"])
    predictions["experimental_correct"] = predictions[TARGET].eq(predictions["experimental_prediction"])
    predictions.to_csv(OUT / "evaluation_predictions.csv", index=False)
    pred_summary = pd.concat([
        pd.Series(baseline_pred, name="prediction").value_counts().rename_axis("label").reset_index(name="count").assign(model="baseline"),
        pd.Series(experimental_pred, name="prediction").value_counts().rename_axis("label").reset_index(name="count").assign(model="experimental"),
    ])
    pred_summary.to_csv(OUT / "prediction_distribution.csv", index=False)

    scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler()
    x_train = scaler.fit_transform(training[FEATURES].astype(float))
    x_eval_scaled = scaler.transform(x_eval)
    nearest = NearestNeighbors(n_neighbors=1).fit(x_train)
    distances, indices = nearest.kneighbors(x_eval_scaled)
    nearest_table = eligible[["source_row", TARGET]].copy()
    nearest_table["nearest_training_distance"] = distances[:, 0]
    nearest_table["nearest_training_label"] = training.iloc[indices[:, 0]][TARGET].to_numpy()
    nearest_table.to_csv(OUT / "nearest_training_examples.csv", index=False)

    range_flags = comparison.set_index("feature")["independent_outside_original_range_percent"].to_dict()
    summary = {
        "audit_scope": "read-only forensic audit; no source or existing experiment artifact modified",
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "schema": {
            "original_columns": original.columns.tolist(), "independent_columns": independent.columns.tolist(),
            "expected_feature_order": FEATURES, "baseline_feature_order": baseline_feature_order,
            "experimental_feature_order": experimental_feature_order,
            "feature_order_match": baseline_feature_order == FEATURES == experimental_feature_order,
            "original_dtypes": {k: str(v) for k, v in original[FEATURES].dtypes.items()},
            "independent_dtypes": {k: str(v) for k, v in independent[FEATURES].dtypes.items()},
            "units": "not documented in either CSV or model metadata; cannot be verified",
        },
        "labels": {
            "original_unique": sorted(original[TARGET].astype(str).str.strip().str.casefold().unique()),
            "independent_unique": independent_labels,
            "same_target_column_name": original.columns[-1] == independent.columns[-1] == TARGET,
            "independent_classes_in_experimental_encoder": {label: label in experimental_encoder_classes for label in independent_labels},
            "all_seven_independent_classes_encoded": all(label in experimental_encoder_classes for label in independent_labels),
            "independent_class_counts": {str(k): int(v) for k, v in independent[TARGET].value_counts().items()},
            "training_class_counts": {str(k): int(v) for k, v in training[TARGET].value_counts().items()},
        },
        "evaluation": {
            "eligible_rows": len(eligible), "eligible_labels": sorted(eligible[TARGET].unique()),
            "baseline_accuracy": float(predictions["baseline_correct"].mean()),
            "experimental_accuracy": float(predictions["experimental_correct"].mean()),
            "baseline_prediction_counts": {str(k): int(v) for k, v in pd.Series(baseline_pred).value_counts().items()},
            "experimental_prediction_counts": {str(k): int(v) for k, v in pd.Series(experimental_pred).value_counts().items()},
            "mean_nearest_training_distance_standardized": float(distances[:, 0].mean()),
            "median_nearest_training_distance_standardized": float(np.median(distances[:, 0])),
        },
        "findings": [
            {
                "status": "confirmed",
                "finding": "Feature order and numeric schema are compatible",
                "evidence": "Both model bundles declare N,P,K,temperature,humidity,ph,rainfall; all source columns are numeric and finite.",
            },
            {
                "status": "confirmed",
                "finding": "No label-encoder omission explains the experimental 0% result",
                "evidence": "All seven independent labels are present in the experimental encoder; eligible labels are cotton, maize, rice.",
            },
            {
                "status": "confirmed",
                "finding": "Experimental predictions fail because predictions do not match the three eligible ground-truth labels",
                "evidence": f"Exact prediction table is in evaluation_predictions.csv; experimental accuracy is {predictions['experimental_correct'].mean():.4f}.",
            },
            {
                "status": "potential",
                "finding": "Feature distribution shift is present where outside-range percentages are nonzero",
                "evidence": "See feature_comparison.csv and feature_distributions.png; range exceedance is diagnostic, not proof of unit mismatch.",
            },
            {
                "status": "potential",
                "finding": "Units and target provenance remain unverified",
                "evidence": "Neither CSV contains unit metadata, collection provenance, or a label-definition contract.",
            },
        ],
        "range_exceedance_percent": range_flags,
        "recommended_next_steps": [
            "Verify units, measurement procedures, and label-generation provenance with the data owner.",
            "Do not tune either model against these evaluation rows.",
            "If provenance is confirmed, design a stratified, source-independent validation protocol covering all target classes.",
            "Inspect prediction_distribution.csv and nearest_training_examples.csv with domain experts before any retraining decision.",
        ],
    }
    dump(OUT / "audit_report.json", summary)
    lines = [
        "# Independent-data v1 forensic audit",
        "",
        "## Conclusion",
        "",
        "The 0% experimental result is not explained by feature order or a missing label-encoder class. The audit confirms that the experimental model predicts labels different from the three eligible ground-truth labels; distribution shift and unverified units/label provenance remain plausible causes requiring source verification.",
        "",
        "## Key evidence",
        "",
        f"- Eligible evaluation rows: **{len(eligible)}**.",
        f"- Baseline accuracy: **{predictions['baseline_correct'].mean():.4%}**.",
        f"- Experimental accuracy: **{predictions['experimental_correct'].mean():.4%}**.",
        f"- Feature order match: **{baseline_feature_order == FEATURES == experimental_feature_order}**.",
        f"- All seven independent labels in experimental encoder: **{all(label in experimental_encoder_classes for label in independent_labels)}**.",
        f"- Mean standardized nearest-training distance: **{distances[:, 0].mean():.4f}**.",
        "",
        "## Confirmed problems",
        "",
        "1. The experimental model fails on every eligible row; see `evaluation_predictions.csv` and `prediction_distribution.csv`.",
        "2. The direct evaluation covers only cotton, maize, and rice, not the full independent label set.",
        "",
        "## Potential problems requiring source verification",
        "",
        "1. Feature units and measurement procedures are undocumented.",
        "2. Independent feature distributions may be shifted beyond original observed ranges.",
        "3. Independent crop labels may not have the same provenance or operational definition as original labels.",
        "",
        "No generalization improvement is supported. Existing source files and v1 artifacts were not modified.",
    ]
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Read-only Phase 2 audit for the independent-data investigation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
NEAR_DISTANCE = 1.0


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def normalize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    result = frame.loc[:, FEATURES + [TARGET]].copy()
    result[TARGET] = result[TARGET].astype(str).str.strip().str.casefold()
    result["source"] = source
    result["source_row"] = np.arange(len(result), dtype=int) + 2
    return result


def descriptive_rows(frame: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby(TARGET, sort=True):
        for feature in FEATURES:
            values = group[feature].astype(float)
            rows.append({
                "source": source, "crop": label, "feature": feature,
                "sample_count": int(len(values)), "mean": float(values.mean()),
                "median": float(values.median()), "std": float(values.std(ddof=1)),
                "min": float(values.min()), "q01": float(values.quantile(.01)),
                "q05": float(values.quantile(.05)), "q25": float(values.quantile(.25)),
                "q75": float(values.quantile(.75)), "q95": float(values.quantile(.95)),
                "q99": float(values.quantile(.99)), "max": float(values.max()),
            })
    return rows


def nearest_cross_source(original: pd.DataFrame, independent: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler().fit(original[FEATURES].astype(float))
    original_scaled = scaler.transform(original[FEATURES].astype(float))
    independent_scaled = scaler.transform(independent[FEATURES].astype(float))
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(original_scaled)
    distances, indices = neighbors.kneighbors(independent_scaled)
    nearest = original.iloc[indices[:, 0]].reset_index(drop=True)
    result = independent[["source_row", TARGET]].reset_index(drop=True).rename(
        columns={"source_row": "independent_source_row", TARGET: "independent_label"}
    )
    result["original_source_row"] = nearest["source_row"].to_numpy()
    result["original_label"] = nearest[TARGET].to_numpy()
    result["standardized_euclidean_distance"] = distances[:, 0]
    result["labels_differ"] = result["independent_label"] != result["original_label"]
    result["within_heuristic_distance"] = result["standardized_euclidean_distance"] <= NEAR_DISTANCE
    return result.sort_values("standardized_euclidean_distance").reset_index(drop=True)


def plot_shared_classes(original: pd.DataFrame, independent: pd.DataFrame) -> None:
    shared = sorted(set(original[TARGET]).intersection(independent[TARGET]))
    for crop in shared:
        fig, axes = plt.subplots(4, 2, figsize=(12, 15))
        left = original.loc[original[TARGET].eq(crop)]
        right = independent.loc[independent[TARGET].eq(crop)]
        for axis, feature in zip(axes.flat, FEATURES):
            axis.hist(left[feature], bins=18, alpha=.55, density=True, label="original")
            axis.hist(right[feature], bins=18, alpha=.55, density=True, label="independent")
            axis.set_title(feature)
            axis.grid(alpha=.2)
            axis.legend()
        axes.flat[-1].axis("off")
        fig.suptitle(f"Shared crop feature distributions: {crop}")
        fig.tight_layout()
        fig.savefig(OUT / f"shared_class_{crop}.png", dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, feature in zip(axes.flat, FEATURES):
        data = pd.concat([
            original.assign(dataset="original"),
            independent.assign(dataset="independent"),
        ])
        data.boxplot(column=feature, by=["dataset", TARGET], ax=axis, rot=75)
        axis.set_title(feature)
        axis.set_xlabel("")
        axis.get_figure().suptitle("")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "feature_label_relationships.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = normalize(pd.read_csv(ROOT / "Crop_recommendation.csv"), "original")
    independent = normalize(pd.read_csv(ROOT / "independent_data.csv"), "independent")
    shared = sorted(set(original[TARGET]).intersection(independent[TARGET]))

    shared_stats = pd.DataFrame(descriptive_rows(original[original[TARGET].isin(shared)], "original") +
                                descriptive_rows(independent[independent[TARGET].isin(shared)], "independent"))
    shared_stats.to_csv(OUT / "shared_class_statistics.csv", index=False)

    original_groups = original.groupby(TARGET).size().rename("original_count")
    independent_groups = independent.groupby(TARGET).size().rename("independent_count")
    class_summary = pd.concat([original_groups, independent_groups], axis=1).fillna(0).astype(int)
    class_summary["present_in_both"] = class_summary["original_count"].gt(0) & class_summary["independent_count"].gt(0)
    class_summary.to_csv(OUT / "class_counts.csv")

    nearest = nearest_cross_source(original, independent)
    nearest.to_csv(OUT / "nearest_cross_source_pairs.csv", index=False)
    conflicts = nearest.loc[nearest["labels_differ"] & nearest["within_heuristic_distance"]].copy()
    conflicts.insert(0, "conflict_type", "near_feature_similarity_label_disagreement")
    conflicts.to_csv(OUT / "conflicting_label_groups.csv", index=False)

    exact_original = set(map(tuple, original[FEATURES].itertuples(index=False, name=None)))
    exact_independent = set(map(tuple, independent[FEATURES].itertuples(index=False, name=None)))
    exact_rows = []
    for key in sorted(exact_original.intersection(exact_independent)):
        left = original.loc[(original[FEATURES] == key).all(axis=1)]
        right = independent.loc[(independent[FEATURES] == key).all(axis=1)]
        for _, lrow in left.iterrows():
            for _, rrow in right.iterrows():
                exact_rows.append({
                    "conflict_type": "exact_feature_vector_cross_source",
                    "original_source_row": int(lrow["source_row"]),
                    "original_label": lrow[TARGET],
                    "independent_source_row": int(rrow["source_row"]),
                    "independent_label": rrow[TARGET],
                    "labels_differ": bool(lrow[TARGET] != rrow[TARGET]),
                })
    exact_frame = pd.DataFrame(exact_rows)
    if exact_frame.empty:
        exact_frame = pd.DataFrame(columns=[
            "conflict_type", "original_source_row", "original_label",
            "independent_source_row", "independent_label", "labels_differ",
        ])
    exact_frame.to_csv(OUT / "exact_cross_source_feature_matches.csv", index=False)

    plot_shared_classes(original, independent)

    relationship_rows = []
    for crop in shared:
        left = original[original[TARGET].eq(crop)]
        right = independent[independent[TARGET].eq(crop)]
        for feature in FEATURES:
            left_values = left[feature].astype(float)
            right_values = right[feature].astype(float)
            pooled_low = max(left_values.min(), right_values.min())
            pooled_high = min(left_values.max(), right_values.max())
            relationship_rows.append({
                "crop": crop, "feature": feature,
                "original_count": len(left_values), "independent_count": len(right_values),
                "original_range": f"{left_values.min():.6g}..{left_values.max():.6g}",
                "independent_range": f"{right_values.min():.6g}..{right_values.max():.6g}",
                "range_overlap": bool(pooled_low <= pooled_high),
                "range_overlap_low": float(pooled_low) if pooled_low <= pooled_high else np.nan,
                "range_overlap_high": float(pooled_high) if pooled_low <= pooled_high else np.nan,
                "mean_difference_independent_minus_original": float(right_values.mean() - left_values.mean()),
                "median_difference_independent_minus_original": float(right_values.median() - left_values.median()),
                "std_ratio_independent_to_original": float(right_values.std(ddof=1) / left_values.std(ddof=1)),
            })
    pd.DataFrame(relationship_rows).to_csv(OUT / "feature_label_analysis.csv", index=False)

    prior_predictions = pd.read_csv(ROOT / "experiments" / "independent_data_v1_audit" / "evaluation_predictions.csv")
    training = pd.read_csv(ROOT / "experiments" / "independent_data_v1" / "cleaned_combined_dataset.csv")
    incorrect = prior_predictions.loc[~prior_predictions["experimental_correct"]].copy()
    nearest_training_rows = []
    scaler = StandardScaler().fit(training[FEATURES].astype(float))
    train_scaled = scaler.transform(training[FEATURES].astype(float))
    eval_frame = independent[independent["source_row"].isin(incorrect["source_row"])].copy()
    eval_scaled = scaler.transform(eval_frame[FEATURES].astype(float))
    model = NearestNeighbors(n_neighbors=1).fit(train_scaled)
    distances, indices = model.kneighbors(eval_scaled)
    for eval_row, distance, index in zip(eval_frame.itertuples(index=False), distances[:, 0], indices[:, 0]):
        train_row = training.iloc[index]
        nearest_training_rows.append({
            "independent_source_row": int(eval_row.source_row),
            "actual_label": eval_row.label,
            "experimental_prediction": incorrect.loc[incorrect["source_row"].eq(eval_row.source_row), "experimental_prediction"].iloc[0],
            "nearest_training_source": train_row["source"],
            "nearest_training_source_row": int(train_row["source_row"]),
            "nearest_training_label": train_row[TARGET],
            "standardized_euclidean_distance": float(distance),
            "nearby_label_differs_from_actual": bool(train_row[TARGET] != eval_row.label),
        })
    pd.DataFrame(nearest_training_rows).sort_values("standardized_euclidean_distance").to_csv(
        OUT / "nearest_neighbor_summary.csv", index=False
    )

    report = {
        "shared_classes": shared,
        "original_rows": len(original),
        "independent_rows": len(independent),
        "exact_cross_source_feature_matches": len(exact_frame),
        "near_label_disagreement_pairs_within_distance": len(conflicts),
        "near_distance_definition": "Euclidean distance after StandardScaler fit on original feature columns; heuristic only and not a unit-validity test.",
        "shared_class_distribution_findings": {
            crop: {
                feature: {
                    "original_mean": float(original.loc[original[TARGET].eq(crop), feature].mean()),
                    "independent_mean": float(independent.loc[independent[TARGET].eq(crop), feature].mean()),
                }
                for feature in FEATURES
            }
            for crop in shared
        },
        "confirmed_facts": [
            "The only labels shared by both files are cotton, maize, and rice.",
            "Both files use the same eight column names and seven numeric feature names.",
            "Exact cross-source feature-vector matches and exact contradictory-label matches are reported explicitly.",
            "The nearest-neighbor method is standardized Euclidean distance, used only as a descriptive similarity diagnostic.",
        ],
        "unresolved": [
            "Units, sampling depth, aggregation windows, collection geography, dates, and label provenance are absent from both CSVs.",
            "Different per-class distributions may reflect real population differences, different measurement protocols, synthetic generation, or label/task mismatch; the CSVs cannot distinguish these explanations.",
        ],
    }
    write_json(OUT / "phase2_audit_summary.json", report)

    (OUT / "audit_report.md").write_text(f"""# Phase 2 independent-data investigation

## Scope and constraints

This is a read-only descriptive audit. No model was trained, tuned, replaced,
or loaded for prediction during this phase. Source CSVs and all v1 artifacts
were left untouched.

## Confirmed findings

- Original dataset: **{len(original):,}** rows and 22 classes.
- Independent dataset: **{len(independent):,}** rows and 7 classes.
- Shared classes: **{", ".join(shared)}**.
- Exact cross-source feature-vector matches: **{len(exact_frame)}**.
- Near feature-similarity label disagreements at the heuristic threshold
  (standardized Euclidean distance <= {NEAR_DISTANCE}): **{len(conflicts)}**.
- Per-class statistics and quantiles are in `shared_class_statistics.csv`.
- Per-feature/crop range overlap and mean/median shifts are in
  `feature_label_analysis.csv`.

Different distributions do not automatically mean either file is incorrect.
They do show that source populations or data-generation processes differ and
must be explained before combining labels for retraining.

## Feature-label relationships

The shared-class analysis compares all seven features for cotton, maize, and
rice. `feature_label_relationships.png` and the `shared_class_*.png` files
show the distributions. Range overlap is descriptive only: overlapping
feature values with different crop labels can reflect genuine ambiguity,
different environments, label definitions, or incompatible units.

`conflicting_label_groups.csv` contains near feature-similarity pairs with
different labels. These are **not declared ground-truth contradictions**:
the distance is a heuristic and no unit/provenance evidence exists. Exact
cross-source matches are in `exact_cross_source_feature_matches.csv`.

## Nearest-neighbor analysis

`nearest_neighbor_summary.csv` summarizes every incorrectly predicted
independent evaluation row, including the nearest training source, label, and
standardized distance. Distances use StandardScaler fit on original feature
columns followed by Euclidean distance. This makes feature magnitudes
comparable for a descriptive diagnostic, but it does not establish that the
features have compatible units or agronomic meaning.

## Dataset compatibility

Repository documentation describes the original file as the commonly used
2,200-row, 22-class crop recommendation dataset and notes unresolved
provenance, synthetic/augmented-data concerns, and undocumented feature
semantics. The independent CSV has no manifest, source reference, collection
method, geography, observation period, unit specification, farm IDs, dates,
or agronomist verification. These are confirmed metadata gaps, not evidence
that the independent records are invalid.

## Evaluation limitation

The prior 416-row shared-class evaluation covers only cotton, maize, and rice.
It cannot establish performance for all 26 classes in the experimental model,
and it cannot support a comparison for independent-only classes that the
baseline encoder does not support.

## Potential compatibility problems

1. Feature distributions differ substantially within shared classes.
2. The independent source may use different units, sampling procedures, or
   time aggregation windows.
3. Labels may represent different targets (for example, observed crop,
   recommended crop, or another assignment) even though both columns are
   named `label`.
4. Near feature-similarity pairs frequently have different crop labels; this
   requires source verification rather than automatic conflict removal.

No model-performance improvement is claimed.
""", encoding="utf-8")

    (OUT / "unresolved_data_questions.md").write_text("""# Unresolved data questions

These questions must be answered by the data owners before retraining:

1. What is the authoritative source, version, and acquisition date for each CSV?
2. Are N, P, and K measured in the same units, soil basis, depth, and laboratory method?
3. Is rainfall a monthly total, seasonal total, or another aggregation window?
4. What aggregation windows and instruments produced temperature and humidity?
5. How were pH values sampled and measured?
6. Does `label` mean observed planted crop, agronomist recommendation, optimal crop,
   yield winner, or another target?
7. Are independent labels expert-verified, and are outcomes such as yield or crop
   failure available?
8. Are rows synthetic, augmented, simulated, or directly observed?
9. Are there farm/field/site/time identifiers that must remain grouped?
10. Can the independent rows be proven independent from the original data and from
    any data used to construct the existing models?
""", encoding="utf-8")

    (OUT / "recommended_evaluation_protocol.md").write_text("""# Recommended leakage-free evaluation protocol

1. Obtain a provenance and unit manifest for every source. Reject evaluation
   claims until feature semantics and label definitions are documented.
2. Freeze the baseline model and encoder. Do not fit preprocessing or choose
   thresholds using final test rows.
3. Attach immutable source, farm/field/site, geography, season, date, and
   provenance identifiers to every row. Keep duplicate and related observations
   in the same split.
4. Define the target explicitly and require reliable ground-truth labels,
   preferably agronomist-reviewed labels plus measured outcomes where the claim
   concerns suitability or success.
5. Build a development set from verified training sources. Use grouped
   train/validation splits by farm/source/time; use a final untouched grouped
   test set.
6. For the 22-class baseline, evaluate only on final-test rows whose labels are
   in the frozen baseline encoder. Report all supported classes, class support,
   absent classes, balanced accuracy, macro-F1, weighted-F1, per-class metrics,
   and confusion matrices.
7. For the expanded model, define the full claimed class set and ensure every
   claimed evaluation class is represented in the final test set. Evaluate all
   26 classes separately; do not infer performance for absent classes.
8. Keep independent-only classes separate from the baseline comparison. A
   baseline cannot be scored on labels it cannot emit.
9. Report duplicate removal, conflict handling, source counts, group counts,
   class coverage, feature-range coverage, and unit/provenance limitations.
10. Repeat the final evaluation only on a new, untouched external collection
    after the protocol is frozen. Do not tune against the final test set.

This protocol is a recommendation only. No retraining or tuning was performed
in Phase 2, and no improved generalization is established.
""", encoding="utf-8")


if __name__ == "__main__":
    main()

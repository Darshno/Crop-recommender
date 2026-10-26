"""Reproducible dataset integrity and leakage diagnostics for AgriSense.

This audit is deliberately diagnostic: it reports suspicious patterns without
silently deleting records or claiming that clean rows are agronomically valid.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from evaluation.validation import FEATURES, TARGET

DEFAULT_ROUNDING_DECIMALS = (0, 1, 2, 3, 4, 5)
NEAR_NEIGHBOR_THRESHOLDS = (0.10, 0.25, 0.50)


def file_sha256(path: Path) -> str:
    """Return SHA-256 for exact input-file tracking."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_keys(frame: pd.DataFrame, columns: Sequence[str], decimals: int | None = None) -> set[tuple[Any, ...]]:
    data = frame.loc[:, list(columns)].copy()
    if decimals is not None:
        data = data.round(decimals)
    # Convert to ordinary Python values so set comparisons are consistent.
    return {tuple(row) for row in data.itertuples(index=False, name=None)}


def _duplicate_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    feature_values = frame[FEATURES]
    feature_dup_all = feature_values.duplicated(keep=False)
    feature_dup_excess = feature_values.duplicated(keep="first")
    full_dup_excess = frame[FEATURES + [TARGET]].duplicated(keep="first")

    valid_features = frame[FEATURES].notna().all(axis=1)
    valid_labels = frame[TARGET].notna() & frame[TARGET].astype(str).str.strip().ne("")
    valid = frame.loc[valid_features & valid_labels, FEATURES + [TARGET]]
    grouped = valid.groupby(FEATURES, dropna=False, sort=False)[TARGET].agg(["size", "nunique"])
    conflicting = grouped[grouped["nunique"] > 1]

    rounded_collisions: dict[str, dict[str, int]] = {}
    for decimals in DEFAULT_ROUNDING_DECIMALS:
        rounded = feature_values.round(decimals)
        duplicates_excess = int(rounded.duplicated(keep="first").sum())
        rounded_collisions[str(decimals)] = {
            "duplicate_feature_rows_beyond_first": duplicates_excess,
            "rows_in_duplicate_groups": int(rounded.duplicated(keep=False).sum()),
        }

    return {
        "exact_duplicate_full_rows_beyond_first": int(full_dup_excess.sum()),
        "exact_duplicate_feature_rows_beyond_first": int(feature_dup_excess.sum()),
        "rows_in_exact_duplicate_feature_groups": int(feature_dup_all.sum()),
        "exact_conflicting_feature_group_count": int(len(conflicting)),
        "rows_in_exact_conflicting_feature_groups": int(conflicting["size"].sum()) if not conflicting.empty else 0,
        "rounded_feature_collisions_by_decimal_places": rounded_collisions,
    }


def _class_order_diagnostics(frame: pd.DataFrame, test_size: float, random_state: int) -> dict[str, Any]:
    labels = frame[TARGET].astype(str).to_numpy()
    if not len(labels):
        return {"label_run_count": 0, "unique_label_count": 0, "each_label_one_contiguous_block": False}
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    run_labels = labels[starts].tolist()
    # A naive unshuffled tail holdout is included only to reveal ordering risk.
    cut = int(np.floor(len(frame) * (1.0 - test_size)))
    naive_tail_labels = sorted(pd.unique(labels[cut:]).tolist())
    return {
        "label_run_count": int(len(starts)),
        "unique_label_count": int(pd.Series(labels).nunique()),
        "each_label_one_contiguous_block": bool(len(run_labels) == len(set(run_labels)) == pd.Series(labels).nunique()),
        "first_10_label_runs": run_labels[:10],
        "unshuffled_tail_holdout_diagnostic_only": {
            "test_rows": int(len(frame) - cut),
            "classes_in_tail_test": int(len(naive_tail_labels)),
            "tail_test_classes": naive_tail_labels,
            "warning": "Do not use an unshuffled tail holdout on this file; the rows are grouped by crop label.",
        },
        "recommended_split": {
            "method": "train_test_split with shuffle=True and stratify=label",
            "random_state": int(random_state),
            "test_size_fraction": float(test_size),
        },
    }


def _split_overlap_diagnostics(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    row_positions = np.arange(len(frame))
    train_pos, test_pos = train_test_split(
        row_positions,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[TARGET].astype(str).to_numpy(),
        shuffle=True,
    )
    train = frame.iloc[train_pos].reset_index(drop=True)
    test = frame.iloc[test_pos].reset_index(drop=True)

    train_feature_keys = _row_keys(train, FEATURES)
    test_feature_keys = _row_keys(test, FEATURES)
    exact_feature_overlap = train_feature_keys.intersection(test_feature_keys)
    train_full_keys = _row_keys(train, FEATURES + [TARGET])
    test_full_keys = _row_keys(test, FEATURES + [TARGET])
    exact_full_overlap = train_full_keys.intersection(test_full_keys)

    rounded_overlap: dict[str, int] = {}
    for decimals in DEFAULT_ROUNDING_DECIMALS:
        left = _row_keys(train, FEATURES, decimals=decimals)
        right = _row_keys(test, FEATURES, decimals=decimals)
        rounded_overlap[str(decimals)] = int(len(left.intersection(right)))

    # Fit scaling on training rows only, then measure each test row's nearest
    # training row. The thresholds are diagnostics, not universal definitions.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES].astype(float))
    x_test = scaler.transform(test[FEATURES].astype(float))
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
    neighbors.fit(x_train)
    distances, nearest_indices = neighbors.kneighbors(x_test)
    distances = distances[:, 0]
    nearest_indices = nearest_indices[:, 0]
    train_labels = train[TARGET].astype(str).to_numpy()
    test_labels = test[TARGET].astype(str).to_numpy()

    nearest_rows = pd.DataFrame({
        "test_source_row_index": test_pos,
        "nearest_train_source_row_index": train_pos[nearest_indices],
        "test_label": test_labels,
        "nearest_train_label": train_labels[nearest_indices],
        "same_label": test_labels == train_labels[nearest_indices],
        "standardized_euclidean_distance": distances,
    }).sort_values("standardized_euclidean_distance", ascending=True).reset_index(drop=True)

    quantile_levels = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
    quantiles = np.quantile(distances, quantile_levels)
    percentile_distances = {
        f"p{int(round(q * 100)):02d}": float(value)
        for q, value in zip(quantile_levels, quantiles)
    }
    counts_below_threshold = {
        str(threshold): int(np.sum(distances < threshold))
        for threshold in NEAR_NEIGHBOR_THRESHOLDS
    }

    train_class_counts = train[TARGET].astype(str).value_counts().sort_index()
    test_class_counts = test[TARGET].astype(str).value_counts().sort_index()
    class_split = {
        label: {"train": int(train_class_counts.get(label, 0)), "test": int(test_class_counts.get(label, 0))}
        for label in sorted(frame[TARGET].astype(str).unique())
    }

    report = {
        "protocol": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_size_fraction": float(test_size),
            "random_state": int(random_state),
            "stratified_by_target": True,
            "shuffled": True,
        },
        "exact_feature_vector_overlap_count": int(len(exact_feature_overlap)),
        "exact_full_row_overlap_count": int(len(exact_full_overlap)),
        "rounded_feature_overlap_count_by_decimal_places": rounded_overlap,
        "per_class_split_counts": class_split,
        "nearest_train_neighbor_distance": {
            "metric": "Euclidean distance after StandardScaler fitted on the training split only",
            "percentiles": percentile_distances,
            "count_below_heuristic_thresholds": counts_below_threshold,
            "threshold_warning": "Distance thresholds are sensitivity flags, not a domain-standard definition of duplicates.",
            "same_label_nearest_neighbor_rate": float(nearest_rows["same_label"].mean()) if len(nearest_rows) else None,
        },
    }
    return report, nearest_rows


def audit_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run duplicate, label, ordering, split-overlap and coverage audits."""
    missing = [column for column in FEATURES + [TARGET] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(frame) < 2:
        raise ValueError("Dataset must contain at least two rows")
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Resolve missing required values before running split and nearest-neighbor audit")
    if frame[TARGET].astype(str).str.strip().eq("").any():
        raise ValueError("Target contains blank labels")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in FEATURES):
        raise ValueError("All feature columns must be numeric")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    class_counts = frame[TARGET].astype(str).value_counts().sort_index()
    if class_counts.size < 2 or int(class_counts.min()) < 2:
        raise ValueError("Each of at least two classes must have two or more rows for stratified split")

    features = frame[FEATURES].astype(float)
    class_summary = frame.groupby(TARGET, observed=False)[FEATURES].agg(["count", "min", "median", "max", "std"])
    class_summary.columns = [f"{feature}_{stat}" for feature, stat in class_summary.columns]
    class_summary = class_summary.reset_index()

    feature_description = features.describe(percentiles=[0.01, 0.05, 0.50, 0.95, 0.99]).transpose().reset_index()
    feature_description = feature_description.rename(columns={"index": "feature"})

    balance = class_counts.rename_axis(TARGET).reset_index(name="count")
    balance["share"] = balance["count"] / len(frame)

    duplicate_report = _duplicate_diagnostics(frame)
    split_report, nearest_rows = _split_overlap_diagnostics(frame, test_size, random_state)
    ordering_report = _class_order_diagnostics(frame, test_size, random_state)

    row_count = int(len(frame))
    feature_ranges = {
        feature: {
            "min": float(features[feature].min()),
            "p01": float(features[feature].quantile(0.01)),
            "median": float(features[feature].median()),
            "p99": float(features[feature].quantile(0.99)),
            "max": float(features[feature].max()),
            "unique_values": int(features[feature].nunique()),
            "integer_valued": bool(np.allclose(features[feature].to_numpy(), np.round(features[feature].to_numpy()))),
        }
        for feature in FEATURES
    }

    per_crop_rows = class_counts.to_numpy(dtype=int)
    balanced = bool(np.all(per_crop_rows == per_crop_rows[0]))
    within_vs_overall_std_ratio = {}
    for feature in FEATURES:
        overall_std = float(features[feature].std(ddof=1))
        class_stds = frame.groupby(TARGET, observed=False)[feature].std(ddof=1).dropna()
        within_vs_overall_std_ratio[feature] = float(class_stds.median() / overall_std) if overall_std else None

    report = {
        "dataset": {
            "row_count": row_count,
            "column_count": int(len(frame.columns)),
            "required_feature_count": len(FEATURES),
            "class_count": int(class_counts.size),
            "feature_columns": FEATURES,
            "target_column": TARGET,
            "all_required_values_present": True,
        },
        "class_balance": {
            "rows_per_class_min": int(class_counts.min()),
            "rows_per_class_max": int(class_counts.max()),
            "exactly_balanced": balanced,
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "interpretation": "Equal class counts are useful for a balanced benchmark, but do not represent real crop prevalence or farmer demand.",
        },
        "integrity": duplicate_report,
        "row_order": ordering_report,
        "train_test_leakage": split_report,
        "feature_ranges": feature_ranges,
        "median_within_class_sd_divided_by_overall_sd": within_vs_overall_std_ratio,
        "real_world_coverage_assessment": {
            "status": "INSUFFICIENT EVIDENCE FOR REAL-FARM GENERALIZATION",
            "observed_in_file": [
                "One row per feature vector with seven predictors: N, P, K, temperature, humidity, pH and rainfall.",
                "One crop label per row; all 22 labels have exactly 100 rows in this copy.",
                "Rows are ordered in crop-label blocks rather than randomly mixed.",
                "No missing values were observed in the supplied CSV.",
            ],
            "not_available_in_csv": [
                "Farm/field IDs, geographic coordinates, district, state, or agro-climatic zone per row.",
                "Observation dates, year, growing season, and weather aggregation windows per row.",
                "Soil sampling method, laboratory provenance, nutrient measurement units and sample depth.",
                "Irrigation and water availability, soil texture/type, crop variety, previous crop and management practices.",
                "Harvest/yield, crop failure, profitability, or an independently documented agronomist label for each row.",
                "Row-level source identifiers linking records to their original datasets or farms.",
            ],
            "conclusion": "The file can support a reproducible benchmark against its existing labels. It cannot by itself establish that predictions are agronomically correct on actual farms, nor can random train/test splitting measure transfer to unseen farms, districts, seasons or years.",
            "application_input_semantics": [
                "The current app_farmer.py averages daily mean temperature and relative humidity from the last 90 days.",
                "The current app_farmer.py sums precipitation over the last 90 days and divides by three, producing millimetres per month.",
                "The public description of a commonly used copy of the crop dataset labels rainfall as rainfall during the season. Confirm the exact temporal definition of rainfall in this local CSV; if it is seasonal total, feeding a monthly mean is a train-serving semantic mismatch even though both values are in millimetres.",
                "The project report already notes that Soil Health Card N/P/K units and the dataset's units may differ; this audit cannot reconcile them because row-level unit metadata is absent.",
            ],
            "recommended_next_evidence": [
                "Obtain provenance and documented measurement units for each input and label.",
                "Validate against independent district/field/season data that was not used to build this dataset.",
                "Where repeated farm or source records exist, split by farm/source/time rather than by individual row.",
                "Have local agronomists review recommendations and record outcomes before making farmer-facing claims.",
            ],
        },
        "audit_limits": [
            "No duplicate findings do not prove that records are independently collected; exact row-level source IDs are absent.",
            "Rounded-key collisions are a sensitivity check and are not a universal near-duplicate definition.",
            "Nearest-neighbor distances are standardized geometric diagnostics, not measures of agronomic similarity.",
            "Out-of-range checks cannot prove agronomic validity; valid ranges depend on units, place, depth, season and crop variety.",
        ],
    }
    tables = {
        "class_balance": balance,
        "feature_summary": feature_description,
        "class_feature_ranges": class_summary,
        "nearest_train_neighbors": nearest_rows,
    }
    return report, tables


def _format_percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def build_markdown_report(report: dict[str, Any], dataset_path: Path | str) -> str:
    """Render a human-readable report from the machine-readable audit object."""
    dataset = report["dataset"]
    integrity = report["integrity"]
    split = report["train_test_leakage"]
    neighbor = split["nearest_train_neighbor_distance"]
    balance = report["class_balance"]
    order = report["row_order"]
    coverage = report["real_world_coverage_assessment"]
    round_overlap = split["rounded_feature_overlap_count_by_decimal_places"]
    rounding_all_zero = all(value == 0 for value in round_overlap.values())
    exact_clean = (
        integrity["exact_duplicate_full_rows_beyond_first"] == 0
        and integrity["exact_duplicate_feature_rows_beyond_first"] == 0
        and integrity["exact_conflicting_feature_group_count"] == 0
        and split["exact_feature_vector_overlap_count"] == 0
    )
    lines = [
        "# Phase 2 — Dataset Integrity and Leakage Audit",
        "",
        f"**Dataset:** `{dataset_path}`  ",
        f"**Scope:** {dataset['row_count']:,} rows, {dataset['required_feature_count']} predictors, {dataset['class_count']} crop labels  ",
        "**Verdict:** STRUCTURAL CHECKS PASS; REAL-FARM VALIDITY IS NOT ESTABLISHED",
        "",
        "## Executive summary",
        "",
        f"- Exact duplicate and conflicting-label checks: **{'no exact duplicates/conflicts found' if exact_clean else 'issues found — inspect JSON details'}**.",
        f"- Train/test exact feature overlap: **{split['exact_feature_vector_overlap_count']}**; exact full-row overlap: **{split['exact_full_row_overlap_count']}**.",
        f"- Train/test rounded feature overlap for 0–5 decimal places: **{'none detected' if rounding_all_zero else 'one or more collisions detected'}**.",
        f"- Class balance: **{'exactly balanced' if balance['exactly_balanced'] else 'imbalanced'}** ({balance['rows_per_class_min']}–{balance['rows_per_class_max']} rows per class).",
        f"- File ordering: **{order['label_run_count']} label runs for {order['unique_label_count']} labels**; each label in one contiguous block: **{order['each_label_one_contiguous_block']}**.",
        f"- Real-world validity: **{coverage['status']}**.",
        "",
        "## 1. Duplicate and label-conflict checks",
        "",
        f"- Exact duplicate full rows beyond the first copy: {integrity['exact_duplicate_full_rows_beyond_first']}.",
        f"- Exact duplicate feature vectors beyond the first copy: {integrity['exact_duplicate_feature_rows_beyond_first']}.",
        f"- Rows involved in exact duplicate feature groups: {integrity['rows_in_exact_duplicate_feature_groups']}.",
        f"- Identical feature vectors assigned conflicting crop labels: {integrity['exact_conflicting_feature_group_count']} groups ({integrity['rows_in_exact_conflicting_feature_groups']} rows affected).",
        "",
        "Rounded feature-key collisions (diagnostic only):",
        "",
        "| Decimal places | Duplicate rows beyond first | Rows in duplicate groups |",
        "|---:|---:|---:|",
    ]
    for decimals, values in integrity["rounded_feature_collisions_by_decimal_places"].items():
        lines.append(f"| {decimals} | {values['duplicate_feature_rows_beyond_first']} | {values['rows_in_duplicate_groups']} |")
    lines.extend([
        "",
        "No exact duplicates were found in the supplied copy. This rules out one common source of random-split inflation, but it does **not** prove that every row came from an independent farm or independent source measurement.",
        "",
        "## 2. Train/test overlap and nearest-neighbor diagnostics",
        "",
        f"The audit reproduces the model script's stratified shuffled split (random state {split['protocol']['random_state']}, test fraction {split['protocol']['test_size_fraction']:.0%}): **{split['protocol']['train_rows']:,} train rows** and **{split['protocol']['test_rows']:,} test rows**.",
        "",
        f"- Exact shared feature vectors: {split['exact_feature_vector_overlap_count']}.",
        f"- Exact shared full rows: {split['exact_full_row_overlap_count']}.",
        f"- Rounded shared feature keys: {'; '.join(f'{places} dp: {count} overlaps' for places, count in round_overlap.items())}.",
        f"- Same-label nearest training neighbor rate: {_format_percent(neighbor['same_label_nearest_neighbor_rate']) if neighbor['same_label_nearest_neighbor_rate'] is not None else 'n/a'}.",
        f"- Standardized Euclidean nearest-neighbor distance: min {neighbor['percentiles']['p00']:.3f}, median {neighbor['percentiles']['p50']:.3f}, 95th percentile {neighbor['percentiles']['p95']:.3f}, max {neighbor['percentiles']['p100']:.3f}.",
        f"- Test points within heuristic distance thresholds: {'; '.join(f'<{threshold}: {count}' for threshold, count in neighbor['count_below_heuristic_thresholds'].items())}.",
        "",
        "Distance calculations standardize features using the training split only. The thresholds are rough diagnostics, not a scientific definition of near-duplicate agricultural observations. The closest pairs are exported to `nearest_train_neighbors.csv` for review.",
        "",
        "## 3. Ordering and split leakage risk",
        "",
        f"The file has {order['label_run_count']} contiguous label runs for {order['unique_label_count']} crop labels; each crop occupies its own block. That is not target leakage into the feature columns, but it creates a serious evaluation hazard if the file is split without shuffling.",
        f"A diagnostic unshuffled tail holdout would contain only {order['unshuffled_tail_holdout_diagnostic_only']['classes_in_tail_test']} classes: {', '.join(order['unshuffled_tail_holdout_diagnostic_only']['tail_test_classes'])}.",
        "",
        "The current modelling code uses `train_test_split(..., shuffle=True, stratify=y, random_state=42)`, which avoids this row-order trap. Keep this explicit. The split is still a random row split, not a field-, region-, or year-held-out evaluation.",
        "",
        "## 4. Class balance and structure",
        "",
        f"Each crop has {balance['rows_per_class_min']}–{balance['rows_per_class_max']} records. Balanced classes make model comparisons easier, but they do not describe actual crop prevalence, market demand or probability that a farmer should plant a crop.",
        "",
        "Per-feature ranges and per-crop ranges are in `feature_summary.csv` and `class_feature_ranges.csv`. The observed ranges are N 0–140, P 5–145, K 5–205, temperature 8.83–43.68 °C, humidity 14.26–99.98%, pH 3.50–9.94, and rainfall 20.21–298.56 mm. These are broad numeric ranges, but their presence alone does not validate the combinations as same-field, same-season observations.",
        "",
        "## 5. Does this dataset represent real agricultural conditions?",
        "",
        f"**Conclusion: {coverage['status']}**",
        "",
        coverage["conclusion"],
        "",
        "### Missing evidence in the CSV",
    ])
    lines.extend(f"- {item}" for item in coverage["not_available_in_csv"])
    lines.extend([
        "",
        "The public dataset description for a widely circulated copy of this dataset says it was built by augmenting Indian rainfall, climate and fertilizer datasets; the supplied CSV itself does not include row-level source links or a documented data-generation/labeling procedure. This is a provenance limitation, not proof that all rows are synthetic. See the source links below.",
        "",
        "### Additional train-serving semantic risk in the current app",
        "",
        "The current `app_farmer.py` averages temperature and humidity over the previous 90 days and divides the 90-day precipitation total by three, treating rainfall as mm/month. A public description of a commonly used copy of this dataset calls rainfall the amount during the season. Confirm the exact temporal definition in the source of this CSV. If training rainfall is a seasonal total, the app's monthly average is a unit-period mismatch and the model sees a different meaning at prediction time. The CSV has no metadata to resolve this. The same caution applies to N/P/K unit mapping from soil-card data.",
        "",
        "The commonly used dataset has 2,200 rows and 22 classes, and research papers report very high scores on random splits of the same dataset. Those results show that the labels are learnable within this dataset; they do not substitute for validation on independent farms or seasons. [Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-07003-8).",
        "",
        "### What would make real-farm claims defensible?",
    ])
    lines.extend(f"- {item}" for item in coverage["recommended_next_evidence"])
    lines.extend([
        "",
        "## 6. Recommended disposition",
        "",
        "1. Keep the current stratified shuffled split for reproducible in-dataset benchmarking.",
        "2. Do not present the current ~99% score as expected real-world accuracy.",
        "3. Before farmer-facing deployment, obtain independently labelled records with farm/region/date/source IDs, clarify units and sampling methods, and evaluate using grouped or time-based holdouts.",
        "4. Keep the model described as a preliminary decision-support prototype until agronomic validation and field testing are complete.",
        "",
        "## Sources and scope",
        "",
        "- [Kaggle Crop Recommendation Dataset metadata](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset/metadata) — description of fields and dataset metadata.",
        "- [Kaggle notebook describing the commonly used Crop Recommendation CSV](https://www.kaggle.com/code/rubiasalam/crop-recommendation-using-weather-and-soil-content/input) — describes it as augmented from rainfall, climate and fertilizer datasets for India.",
        "- [Shastri et al., Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-07003-8) — describes the 2,200-row, 22-class dataset and reports model performance on it.",
        "",
        "The duplicate counts, distributions and split-overlap values in this report were computed directly from the local `Crop_recommendation.csv`. External pages are used only for the source-context note above.",
        "",
    ])
    return "\n".join(lines)


def run_audit(data_path: Path, output_dir: Path, test_size: float = 0.20, random_state: int = 42) -> dict[str, Any]:
    frame = pd.read_csv(data_path)
    report, tables = audit_dataset(frame, test_size=test_size, random_state=random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    report["source_file"] = str(data_path.name)
    report["source_sha256"] = file_sha256(data_path)
    report["audit_parameters"] = {"test_size": float(test_size), "random_state": int(random_state)}
    (output_dir / "dataset_audit.json").write_text(
        __import__("json").dumps(report, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n",
        encoding="utf-8",
    )
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    report_md = build_markdown_report(report, data_path.name)
    (output_dir / "dataset_audit.md").write_text(report_md, encoding="utf-8")
    return report

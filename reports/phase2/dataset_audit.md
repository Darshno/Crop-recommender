# Phase 2 — Dataset Integrity and Leakage Audit

**Dataset:** `Crop_recommendation.csv`  
**Scope:** 2,200 rows, 7 predictors, 22 crop labels  
**Verdict:** STRUCTURAL CHECKS PASS; REAL-FARM VALIDITY IS NOT ESTABLISHED

## Executive summary

- Exact duplicate and conflicting-label checks: **no exact duplicates/conflicts found**.
- Train/test exact feature overlap: **0**; exact full-row overlap: **0**.
- Train/test rounded feature overlap for 0–5 decimal places: **none detected**.
- Class balance: **exactly balanced** (100–100 rows per class).
- File ordering: **22 label runs for 22 labels**; each label in one contiguous block: **True**.
- Real-world validity: **INSUFFICIENT EVIDENCE FOR REAL-FARM GENERALIZATION**.

## 1. Duplicate and label-conflict checks

- Exact duplicate full rows beyond the first copy: 0.
- Exact duplicate feature vectors beyond the first copy: 0.
- Rows involved in exact duplicate feature groups: 0.
- Identical feature vectors assigned conflicting crop labels: 0 groups (0 rows affected).

Rounded feature-key collisions (diagnostic only):

| Decimal places | Duplicate rows beyond first | Rows in duplicate groups |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |
| 4 | 0 | 0 |
| 5 | 0 | 0 |

No exact duplicates were found in the supplied copy. This rules out one common source of random-split inflation, but it does **not** prove that every row came from an independent farm or independent source measurement.

## 2. Train/test overlap and nearest-neighbor diagnostics

The audit reproduces the model script's stratified shuffled split (random state 42, test fraction 20%): **1,760 train rows** and **440 test rows**.

- Exact shared feature vectors: 0.
- Exact shared full rows: 0.
- Rounded shared feature keys: 0 dp: 0 overlaps; 1 dp: 0 overlaps; 2 dp: 0 overlaps; 3 dp: 0 overlaps; 4 dp: 0 overlaps; 5 dp: 0 overlaps.
- Same-label nearest training neighbor rate: 97.95%.
- Standardized Euclidean nearest-neighbor distance: min 0.105, median 0.395, 95th percentile 0.701, max 1.145.
- Test points within heuristic distance thresholds: <0.1: 0; <0.25: 49; <0.5: 325.

Distance calculations standardize features using the training split only. The thresholds are rough diagnostics, not a scientific definition of near-duplicate agricultural observations. The closest pairs are exported to `nearest_train_neighbors.csv` for review.

## 3. Ordering and split leakage risk

The file has 22 contiguous label runs for 22 crop labels; each crop occupies its own block. That is not target leakage into the feature columns, but it creates a serious evaluation hazard if the file is split without shuffling.
A diagnostic unshuffled tail holdout would contain only 5 classes: coconut, coffee, cotton, jute, papaya.

The current modelling code uses `train_test_split(..., shuffle=True, stratify=y, random_state=42)`, which avoids this row-order trap. Keep this explicit. The split is still a random row split, not a field-, region-, or year-held-out evaluation.

## 4. Class balance and structure

Each crop has 100–100 records. Balanced classes make model comparisons easier, but they do not describe actual crop prevalence, market demand or probability that a farmer should plant a crop.

Per-feature ranges and per-crop ranges are in `feature_summary.csv` and `class_feature_ranges.csv`. The observed ranges are N 0–140, P 5–145, K 5–205, temperature 8.83–43.68 °C, humidity 14.26–99.98%, pH 3.50–9.94, and rainfall 20.21–298.56 mm. These are broad numeric ranges, but their presence alone does not validate the combinations as same-field, same-season observations.

## 5. Does this dataset represent real agricultural conditions?

**Conclusion: INSUFFICIENT EVIDENCE FOR REAL-FARM GENERALIZATION**

The file can support a reproducible benchmark against its existing labels. It cannot by itself establish that predictions are agronomically correct on actual farms, nor can random train/test splitting measure transfer to unseen farms, districts, seasons or years.

### Missing evidence in the CSV
- Farm/field IDs, geographic coordinates, district, state, or agro-climatic zone per row.
- Observation dates, year, growing season, and weather aggregation windows per row.
- Soil sampling method, laboratory provenance, nutrient measurement units and sample depth.
- Irrigation and water availability, soil texture/type, crop variety, previous crop and management practices.
- Harvest/yield, crop failure, profitability, or an independently documented agronomist label for each row.
- Row-level source identifiers linking records to their original datasets or farms.

The public dataset description for a widely circulated copy of this dataset says it was built by augmenting Indian rainfall, climate and fertilizer datasets; the supplied CSV itself does not include row-level source links or a documented data-generation/labeling procedure. This is a provenance limitation, not proof that all rows are synthetic. See the source links below.

### Additional train-serving semantic risk in the current app

The current `app_farmer.py` averages temperature and humidity over the previous 90 days and divides the 90-day precipitation total by three, treating rainfall as mm/month. A public description of a commonly used copy of this dataset calls rainfall the amount during the season. Confirm the exact temporal definition in the source of this CSV. If training rainfall is a seasonal total, the app's monthly average is a unit-period mismatch and the model sees a different meaning at prediction time. The CSV has no metadata to resolve this. The same caution applies to N/P/K unit mapping from soil-card data.

The commonly used dataset has 2,200 rows and 22 classes, and research papers report very high scores on random splits of the same dataset. Those results show that the labels are learnable within this dataset; they do not substitute for validation on independent farms or seasons. [Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-07003-8).

### What would make real-farm claims defensible?
- Obtain provenance and documented measurement units for each input and label.
- Validate against independent district/field/season data that was not used to build this dataset.
- Where repeated farm or source records exist, split by farm/source/time rather than by individual row.
- Have local agronomists review recommendations and record outcomes before making farmer-facing claims.

## 6. Recommended disposition

1. Keep the current stratified shuffled split for reproducible in-dataset benchmarking.
2. Do not present the current ~99% score as expected real-world accuracy.
3. Before farmer-facing deployment, obtain independently labelled records with farm/region/date/source IDs, clarify units and sampling methods, and evaluate using grouped or time-based holdouts.
4. Keep the model described as a preliminary decision-support prototype until agronomic validation and field testing are complete.

## Sources and scope

- [Kaggle Crop Recommendation Dataset metadata](https://www.kaggle.com/datasets/siddharthss/crop-recommendation-dataset/metadata) — description of fields and dataset metadata.
- [Kaggle notebook describing the commonly used Crop Recommendation CSV](https://www.kaggle.com/code/rubiasalam/crop-recommendation-using-weather-and-soil-content/input) — describes it as augmented from rainfall, climate and fertilizer datasets for India.
- [Shastri et al., Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-07003-8) — describes the 2,200-row, 22-class dataset and reports model performance on it.

The duplicate counts, distributions and split-overlap values in this report were computed directly from the local `Crop_recommendation.csv`. External pages are used only for the source-context note above.

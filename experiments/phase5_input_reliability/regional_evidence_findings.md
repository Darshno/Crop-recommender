# Phase 5 Regional Evidence Findings

## Source and coverage

### Confirmed by source code or data

- The source is [data/regional/crop_yield.csv](C:/Users/darsh/Downloads/agrisense_phase1/data/regional/crop_yield.csv).
- The repository inventory found 19,689 rows, 30 states, 55 crop names, six
  seasons, and years 1997–2020.
- Required columns are `Crop`, `Crop_Year`, `Season`, `State`, `Area`,
  `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`, and `Yield`.
- `load_regional_data()` in [regional_recommender.py](C:/Users/darsh/Downloads/agrisense_phase1/regional_recommender.py)
  records the source path, validates required columns, normalizes text,
  coerces numeric columns, and drops rows with invalid/non-finite values.

### Inferred

- `Yield` may be derived from production and area, but this cannot be treated
  as fact without the source data dictionary.

### Unresolved

- Original publisher, attribution/licensing, collection method, production
  units, yield units, area units, missingness meaning, and completeness by
  state/crop/season/year.

## Filtering and aggregation

### Confirmed by source code or data

- `build_historical_evidence()` normalizes state and season matching keys,
  requires `Crop_Year < cutoff_year`, and filters the selected state/season.
- It aggregates median `Yield`, observation count, and last observed year by
  crop.
- `Production` is loaded and validated but is not used for ranking.
- `rank_with_regional_evidence()` ignores evidence below its configured
  observation-support rule, scales median yield within the selected evidence
  set, and returns a fallback when no usable evidence remains.
- Crop names not present in the classifier labels are returned separately as
  regional-only candidates.
- `evaluate_phase4.py` uses prior years only for chronological evaluation.

### Inferred

- A zero production/yield value is currently treated as a valid numeric
  observation, while missing/non-numeric values are dropped. Whether zero means
  true zero, suppressed data, or no cultivation requires source verification.

### Unresolved

- Whether state/season/crop normalization can merge distinct source categories.
- Whether observation count means independent farms, administrative records, or
  repeated aggregate records.
- Whether median yield is comparable across crops, years, and regions.

## Separation from classifier

### Confirmed by source code or data

- The regional CSV is not concatenated with `Crop_recommendation.csv`.
- It does not provide the seven soil-model features.
- The classifier produces the base ranking; regional evidence is applied in a
  separate presentation-layer hybrid ranking.
- The UI describes historical evidence as separate and not proof of suitability.

### Inferred

- Even a separate hybrid ranking can be interpreted by users as a suitability
  score unless the UI distinguishes model ranking, historical observation, and
  unsupported inference.

## Transparent UI proposal

Show, for the selected state/season/year:

1. Source name and attribution status.
2. Coverage range and latest pre-cutoff year.
3. Matching crop name and number of observations.
4. Median yield with “unit not documented” until verified.
5. Explicit distinction between missing records and numeric zero.
6. “No historical evidence found” for no rows; never convert absence to zero.
7. Classifier ranking and historical evidence in separate panels.
8. No unsupported “regional suitability score,” yield forecast, or causal claim.


# Regional Evidence Audit

## Dataset

### Confirmed from source code or data

- [`data/regional/crop_yield.csv`](C:/Users/darsh/Downloads/agrisense_phase1/data/regional/crop_yield.csv)
  has 19,689 rows, 30 states, 55 crop names, six seasons, and years 1997–2020.
  Columns include `Crop`, `Crop_Year`, `Season`, `State`, `Area`,
  `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`, and `Yield`.
- `load_regional_data()` validates required columns, trims/case-normalizes
  crop/state/season text, coerces numeric fields, and drops invalid rows while
  recording a quality report.
- `build_historical_evidence()` filters exact normalized state and season and
  uses `Crop_Year < cutoff_year`, then summarizes median `Yield`, observation
  count, and last year. It does not use raw `Production` for ranking.
- `rank_with_regional_evidence()` normalizes median yield within the selected
  evidence set and combines it with the classifier score using default weight
  0.20 and minimum two observations. Sparse/missing evidence returns the
  unchanged base ranking.
- Regional crop names absent from the model's 22 encoder classes are returned
  as `regional_only_candidates`, not claimed as classifier predictions.
- `evaluate_phase4.py` performs chronological year-held-out evaluation using
  only earlier years and reports Recall@3/NDCG@3, support, fallback coverage,
  and per-state summaries.
- The UI labels the section “Historical regional evidence” and states that it
  is separate from the soil-model prediction and not proof of suitability.

### Inferred and requiring verification

- The `Yield` field may be a derived production/area measure, but its unit and
  denominator are not documented in this repository. It must not be displayed
  with an invented unit.
- A zero `Production` or `Yield` is currently a numeric observation, while a
  missing/invalid row is dropped. This distinction should be confirmed with
  the source data dictionary before interpretation.

### Unresolved

- Source attribution, licensing, collection methodology, missingness meaning,
  production/yield units, and whether records are complete for each
  state/crop/season/year are unresolved.
- Historical association does not establish crop suitability, causality,
  expected yield, or benefit from following a recommendation.

## Separation assessment

The regional module does not concatenate the datasets or feed regional
features into the soil classifier. It is kept separate until the UI combines
scores for presentation. This is a defensible boundary, but the hybrid score
still changes ranking and therefore needs explicit coverage, provenance, and
uncertainty display.


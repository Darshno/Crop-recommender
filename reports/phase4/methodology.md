# Phase 4 — Regional evidence methodology

## Scope

Phase 4 adds historical regional evidence without changing `crop_model.pkl` or
`Crop_recommendation.csv`. The soil model remains the primary prediction and
its 22 trained classes are never expanded or relabelled.

The regional source is `data/regional/crop_yield.csv`. It contains
crop/state/season/year production records, but it does not contain soil N, P,
K, pH, temperature, humidity or rainfall measurements compatible with the
soil model. The two datasets therefore are not concatenated.

## Data handling

`regional_recommender.py` trims and case-normalizes crop, state and season
matching keys, validates all required columns, converts numeric fields with
non-finite/malformed values to invalid rows, drops those rows, and writes a
quality report during evaluation. The source does not provide authoritative
yield units, so all displays label yield as **dataset-dependent**.

For a selected state, season and evaluation year, evidence is built only from
rows with `Crop_Year < evaluation_year`. Each crop is summarized with median
yield and observation count. Raw production is not used for ranking because it
is confounded by cultivated area.

## Ranking and fallback

The unchanged soil-model probabilities are returned as the base ranking.
Regional evidence is separately scaled within the selected state/season and
combined with the base scores using a configurable conservative default weight
of 0.20. Evidence with fewer than two observations is ignored by the UI
ranking. Missing or sparse evidence returns the exact base ordering.

Regional crops whose normalized names do not occur in the soil encoder are
reported as historical candidates only. They are not presented as soil-model
predictions.

## Chronological evaluation

`evaluate_phase4.py` evaluates each held-out year using evidence from earlier
years only. It reports Recall@3, NDCG@3, support and coverage overall and in
the row-level CSV by state/season/year, plus state-level aggregates in
`per_state_results.csv`. Because the regional dataset lacks
soil features, the base comparator is a uniform frozen-model-class ranking;
this is a limitation, not a soil prediction. Rows are never randomly split,
and held-out rows are not used to build their evidence.

These metrics measure historical association/ranking behavior, not crop
suitability, causality, yield improvement or validated recommendation
accuracy.

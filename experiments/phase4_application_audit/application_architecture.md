# AgriSense Phase 4 Application Architecture Audit

## Scope and evidence

This is a read-only source audit. No production source, model pickle, CSV, or
previous experiment artifact was changed. Evidence was taken from the files
linked below and from the existing CSV headers/row counts.

## Entry points and flow

### Confirmed from source code or data

- [`app.py`](C:/Users/darsh/Downloads/agrisense_phase1/app.py) is a general
  Streamlit entry point. `load()` loads `crop_model.pkl`; the page collects
  soil, weather, and optional regional context, calls
  `validate_input_values`, then calls `model.predict_proba`.
- [`app_farmer.py`](C:/Users/darsh/Downloads/agrisense_phase1/app_farmer.py)
  is a second, multilingual/voice-oriented Streamlit entry point. Its
  `load()` function loads the same pickle, `fill_weather()` obtains weather,
  `scan_card()`/`parse_lines()` process soil-card images, and the `result`
  screen validates and predicts.
- [`crop_model.py`](C:/Users/darsh/Downloads/agrisense_phase1/crop_model.py)
  is the training/evaluation/recommendation module. `train_and_evaluate()`
  performs the split, model selection, calibration, evaluation, and bundle
  writing. `recommend()` is a programmatic inference entry point.
- The shared feature contract is defined as
  `["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]` in
  [`evaluation/validation.py`](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/validation.py).
  Both apps build a row with `dict(zip(FEATURES, ...))` and then construct a
  DataFrame in that order.
- The pickle bundle contains `model`, `encoder`, `features`,
  `training_ranges`, and metadata, as written in
  [`crop_model.py`](C:/Users/darsh/Downloads/agrisense_phase1/crop_model.py).
- [`regional_recommender.py`](C:/Users/darsh/Downloads/agrisense_phase1/regional_recommender.py)
  is a separate historical evidence layer. It normalizes text, filters
  historical rows, aggregates median yield and observation count, and merges
  an evidence score with classifier scores only at presentation time.

### Inferred and requiring verification

- The intended runtime architecture is two alternative Streamlit UIs over one
  frozen model bundle, not a service with a separately versioned prediction
  API. This should be confirmed before implementation work introduces a shared
  serving layer.
- Relative paths in `app_farmer.py` and in its CSV load are dependent on the
  process working directory, whereas `app.py` uses a resolved path for the
  model. Deployment launch-directory behavior should be verified.

### Unresolved

- The original units, sampling method, and aggregation period for every
  training feature are not documented in the repository. The input contract
  in [`evaluation/phase3.py`](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/phase3.py)
  intentionally leaves these semantics unresolved.
- No source-of-truth data dictionary or provenance record for
  `Crop_recommendation.csv` was found.

## Explanations and observability

### Confirmed

- No SHAP computation or local explanation function is present in application
  code. The repository contains SHAP-named image artifacts, but source search
  found no SHAP runtime integration.
- Training writes global tree/linear feature importance in
  `feature_importance.csv` through `_feature_importances()` and a plot through
  `_generate_plots()`. This is a model diagnostic, not an agronomic explanation.
- The UI labels output as a model score and explicitly says it is not crop
  success probability or expected yield.

## Existing validation boundaries

### Confirmed

- `validate_input_values()` rejects missing, non-numeric, non-finite, and
  broad-bound violations, and warns for values outside training min/max or
  1st/99th percentiles.
- `validate_dataset()` checks required schema, numeric coercion, finite values,
  broad bounds, labels, duplicate rows, and conflicting duplicate feature
  vectors.
- The existing tests are concentrated in
  [`tests/`](C:/Users/darsh/Downloads/agrisense_phase1/tests/) and cover
  dataset audit, validation, metrics, recommendation contract, phase-3
  provenance, and regional filtering. They do not execute either Streamlit UI
  end to end.


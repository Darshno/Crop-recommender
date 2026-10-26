# Rainfall Pipeline Audit

## Trace

### Confirmed from source code or data

1. `app.py: get_weather()` calls Open-Meteo archive API for daily
   `precipitation_sum`, `temperature_2m_mean`, and
   `relative_humidity_2m_mean`, with `start_date` 95 days before the current
   date and `end_date` five days before the current date.
2. It sums daily precipitation and divides by `(days / 30)`; with the default
   90 days this is `/3`, displayed as an average mm/month.
3. `app_farmer.py: fill_weather()` similarly requests a 90-day archive window,
   sums `precipitation_sum`, divides by `3`, and clamps the displayed/input
   value to 20–300. Temperature and humidity are averaged and clamped too.
4. The prediction row uses the resulting value as the seventh feature
   `rainfall`, in the order defined by `FEATURES`.
5. `Crop_recommendation.csv` contains a `rainfall` column, but no period,
   source, unit definition, or aggregation metadata is present in the
   repository.
6. [`evaluation/phase3.py`](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/phase3.py)
   explicitly requires an external manifest to state whether rainfall is a
   monthly or growing-season total and marks the input contract unresolved.

### Inferred and requiring verification

- Open-Meteo `precipitation_sum` is intended as daily precipitation in mm, but
  the exact archive variable semantics and missing-day behavior should be
  verified against the API response metadata for production.
- The UI's `/3` is a 90-day-to-30-day average approximation, not a
  measurement of the training feature.

### Unresolved

- Compatibility between the app's average monthly rainfall and the training
  dataset's `rainfall` feature cannot be established from this repository.
  It could represent monthly, seasonal, annual, or another aggregation.
- No crop-season alignment, location-specific historical period, missing-day
  policy, or uncertainty interval is passed to the model.

## Risk

The current UI silently supplies a value with a declared label (“mm/month”)
while the model semantic is unknown. This is a high-risk validity issue even
when the numeric value is physically plausible. Do not silently change the
feature or retrain until the original training-data definition is recovered.

## Implementation requirements

- Record source, coordinates, requested interval, returned coverage, units, and
  aggregation in the prediction context.
- Block or clearly mark predictions when the training rainfall period is not
  verified; do not convert 90-day totals by assumption.
- Add tests for 90/89/partial-day responses, missing precipitation, API errors,
  clamping, and feature-order preservation.


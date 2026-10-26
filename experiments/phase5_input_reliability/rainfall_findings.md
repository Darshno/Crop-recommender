# Phase 5 Rainfall Input Semantics Findings

## Trace from acquisition to model

### Confirmed by source code or data

| Stage | `app.py` | `app_farmer.py` | Status |
|---|---|---|---|
| Source | Open-Meteo geocoding then archive API | District/town coordinates, then Open-Meteo archive API | Confirmed |
| API variable | Daily `precipitation_sum` plus daily mean temperature/humidity | Same | Confirmed |
| Date interval | `end = today - 5 days`; `start = end - 90 days` | Same effective 90-day window | Confirmed |
| Aggregation | Sum daily precipitation | Sum daily precipitation | Confirmed |
| Transformation | Divide by `(days / 30)`, default `/3` | Divide by `3` | Confirmed |
| Display/input label | `mm/month`, “last 90 days” | `mm` in result chip; same transformed value | Confirmed |
| Clamping | 20–300 for rainfall before prediction | 20–300 for rainfall before prediction | Confirmed |
| Model field | Seventh feature, `rainfall` | Seventh feature, `rainfall` | Confirmed |

The relevant functions are `get_weather()` and `fill_weather()` in
[app.py](C:/Users/darsh/Downloads/agrisense_phase1/app.py) and
[app_farmer.py](C:/Users/darsh/Downloads/agrisense_phase1/app_farmer.py).

## Semantics assessment

### Confirmed by source code or data

- The training CSV contains a `rainfall` column but does not contain a period,
  source, coordinate, or aggregation metadata.
- The Phase 3 input contract in [evaluation/phase3.py](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/phase3.py)
  requires the rainfall period to be explicitly documented and marks it
  unresolved otherwise.
- The current `/3` operation produces an average monthly-equivalent value from
  a roughly 90-day total. It is not a demonstrated conversion to the
  training-feature meaning.
- The app clamps the transformed value, which can hide the magnitude of the
  original observation unless the raw value is retained.

### Inferred

- Open-Meteo's daily precipitation variable is intended to be millimetres, but
  production should retain and verify API metadata, returned dates, and missing
  values rather than relying only on the variable name.
- The current UI presents the approximation as usable model input despite the
  unknown training period; this is a semantic-risk path rather than a numeric
  validation failure.

### Unresolved

- Whether training rainfall is monthly, seasonal, annual, growing-season,
  station-derived, or another aggregation.
- Whether the training value represents the same geography and crop season as
  the current location/weather window.
- Missing-day handling, API coverage completeness, and uncertainty.
- Whether the five-day archive lag is suitable for the intended use.

## User-facing warning proposal

Until the training definition is verified, display a blocking or prominent
warning such as:

> “Weather rainfall is an average derived from the last 90 days. The training
> data does not document whether its rainfall value is monthly, seasonal, or
> another period. Compatibility is unverified, so this result is only a model
> experiment and should not be used as agronomic advice.”

The implementation must not silently substitute a different period or invent a
conversion. Any future change should retain source coordinates, requested and
returned dates, daily-row count, raw total, transformation, clamped value, and
semantic status.

## Tests needed for implementation

- Exact date-window and expected-day-count behavior.
- 90-day, 89-day, and partial/missing-day responses.
- Missing/non-finite precipitation values.
- API error, malformed JSON, empty response, and geocode failure.
- Raw value retained when clamping occurs.
- Feature order and declared rainfall semantics reaching prediction.
- Blocking/warning behavior when training rainfall period remains unresolved.


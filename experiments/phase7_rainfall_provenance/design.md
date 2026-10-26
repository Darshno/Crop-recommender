# Phase 7 Rainfall Provenance Integration Design

## Existing behavior before this change

### `app.py`

`fill_weather()` geocodes a place, requests the Open-Meteo archive endpoint,
requests daily `temperature_2m_mean`, `relative_humidity_2m_mean`, and
`precipitation_sum`, then calls the weather transformation. The rainfall
arithmetic was:

```text
sum(daily precipitation_sum) / (days / 30)
```

with `days=90`, followed by the existing clamp to `[20.0, 300.0]`. The model
received that clamped value as the seventh feature. The legacy `get_weather()`
three-value return contract is preserved.

### `app_farmer.py`

`fill_weather()` requests the same daily archive fields for a 90-day interval,
computes:

```text
sum(daily precipitation_sum) / 3
```

and applies the existing `[20.0, 300.0]` clamp before storing `S.w`. The model
receives the stored rainfall as the seventh feature.

Before Phase 7, neither path retained source, interval, returned-date, missing
value, aggregation, or clamping metadata.

## Shared implementation

Both paths now call:

```text
evaluation.input_semantics.summarize_rainfall_response()
```

The helper preserves the existing pandas sum behavior: missing precipitation
values are skipped by `Series.sum()`, and the divisor is independent of the
number of returned rows. It does not infer a unit or training compatibility.

`app.py` uses `get_weather_with_provenance()` and keeps `get_weather()` as a
legacy wrapper returning `(temperature, humidity, rainfall)`.

`app_farmer.py` uses the same summarizer directly at its inline extraction
boundary.

## Provenance fields retained

`RainfallProvenance` now records:

- provider/source;
- total original precipitation value represented by the returned daily rows;
- original unit, when explicitly supplied;
- requested start/end dates;
- actual first/last returned dates;
- requested interval;
- observation count;
- missing precipitation count;
- missing-data treatment;
- aggregation expression;
- transformed value supplied to the caller;
- transformation history;
- clamping history;
- source metadata such as endpoint;
- explicit training-feature compatibility status.

Raw daily values are not retained. The compact summary is sufficient to audit
the arithmetic and returned coverage without unnecessarily retaining detailed
location/weather observations.

## Compatibility status

The applications do not supply `True` for training-feature compatibility. The
training rainfall aggregation period remains undocumented, so the resulting
semantics context records compatibility as unresolved and adds the existing
Phase 6 unresolved-semantics finding.

This does not mean the API failed or the arithmetic is incorrect. API success,
coverage, arithmetic, units, training-feature compatibility, and agronomic
validity remain separate properties.

## Inference integration boundary

At prediction time both apps construct a Phase 6
`PredictionInputSemantics` context when rainfall provenance is available, then
continue using the existing numeric validation and model row construction.
The model itself still accepts only the seven numeric values and does not
accept metadata. No prediction score or model behavior is changed.

If a user has not fetched weather, no fabricated rainfall provenance is
created; the existing manual/default path remains unchanged.


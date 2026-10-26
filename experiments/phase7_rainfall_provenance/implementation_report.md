# Phase 7 Rainfall Provenance Implementation Report

## Files created

- [experiments/phase7_rainfall_provenance/design.md](C:/Users/darsh/Downloads/agrisense_phase1/experiments/phase7_rainfall_provenance/design.md)
- [experiments/phase7_rainfall_provenance/implementation_report.md](C:/Users/darsh/Downloads/agrisense_phase1/experiments/phase7_rainfall_provenance/implementation_report.md)

## Files modified

- [evaluation/input_semantics.py](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/input_semantics.py)
  - Extended `RainfallProvenance` with compact returned-interval, coverage,
    missing-data, clamping, and source metadata.
  - Added `summarize_rainfall_response()`.
  - Added `record_rainfall_clamp()`.
- [app.py](C:/Users/darsh/Downloads/agrisense_phase1/app.py)
  - Added provenance-aware weather extraction.
  - Preserved the legacy `get_weather()` numeric return shape.
  - Stores rainfall provenance in session state.
  - Builds the Phase 6 semantics context before prediction.
- [app_farmer.py](C:/Users/darsh/Downloads/agrisense_phase1/app_farmer.py)
  - Uses the shared rainfall summarizer.
  - Stores rainfall provenance in session state.
  - Builds the Phase 6 semantics context before prediction.
- [tests/test_input_semantics.py](C:/Users/darsh/Downloads/agrisense_phase1/tests/test_input_semantics.py)
  - Added mocked-data tests for aggregation, missing values, returned dates,
    coverage, transformations, unresolved compatibility, and clamping.

## Exact numeric behavior preservation

The shared helper calculates the same expressions as the existing applications:

- `app.py`: `sum(precipitation_sum) / (days / 30)`, default divisor `3.0`.
- `app_farmer.py`: `sum(precipitation_sum) / 3`.
- Both retain the existing rainfall clamp `[20.0, 300.0]`.
- Missing precipitation values retain pandas’ existing skip-in-sum behavior.
- The model feature order remains `N, P, K, temperature, humidity, ph, rainfall`.

The legacy `app.py:get_weather()` function still returns the same
`(temperature, humidity, rainfall)` shape and delegates to the same arithmetic.

## Tests

Focused Phase 7, Phase 6, and recommendation tests:

```text
15 tests executed during focused validation
```

The final full suite result was:

```text
47 passed, 1 warning
```

The warning is the existing sklearn warning for requesting top-3 accuracy with
exactly three classes.

Python syntax checks with `py_compile` passed for the changed Python modules.

Tests do not access the network. They use mocked pandas weather payloads.

## Both application paths

Both applications now use the shared
`evaluation.input_semantics.summarize_rainfall_response()` implementation:

- `app.py` through `get_weather_with_provenance()`.
- `app_farmer.py` directly inside `fill_weather()`.

The two applications still have separate API-request and geocoding code. They
share the transformation/provenance boundary, not the complete weather-client
implementation.

## Remaining gaps

- The Open-Meteo response metadata is not currently used to populate a unit;
  `original_unit` remains missing in the application path rather than being
  fabricated.
- The training dataset does not document the rainfall aggregation period, so
  `training_feature_compatibility` remains unresolved.
- Daily observations are summarized rather than retained individually.
- The Streamlit UI does not yet render the full provenance context to users.
- The model accepts numeric values only; provenance is retained in the
  application context and is not serialized into the model artifact.
- Manual/default rainfall paths have no weather provenance, intentionally.
- No thresholds, abstention rules, model changes, or agronomic claims were
  introduced.

## Protected-file hash verification

The following protected files were unchanged before and after the
implementation:

| File | SHA-256 |
|---|---|
| `crop_model.pkl` | `27A7AA782ED7A3A6E5C6FFC43BFEDC9AAC54D97CFF71316C812C884C49BC16B6` |
| `Crop_recommendation.csv` | `54A5A6E5408668E668667EFC50DE2FC867C1B875E0431B4F54DD331B0A109A4E` |
| `independent_data.csv` | `2EEB7754870B96D68772471E90BC763E8FCED03D19461158406B45126C20568D` |
| `data/regional/crop_yield.csv` | `AB9BC356B1F8107D490376CEC24450A0E2906322AD25788AB22FB32537AB1F8F` |

Authorized source hashes changed as follows:

| File | Before | After |
|---|---|---|
| `app.py` | `B9C003C516C45181FFF3DF9DF8B795ABBC3BD510669A076E2368999E85205539` | `C58473431568F287F5C7A4DCA2FCD7D54A9506BFCA61E6FDDD84519418D514EB` |
| `app_farmer.py` | `11925BD0D1A20CFE0F3CC48CA045201A642D5E3ABE1A5AB798335B78BA840AF4` | `63A641D737E1803099664301ECA0D80F675EBDBB0B9D0A687E7B3EEF8DD2784D` |
| `evaluation/input_semantics.py` | `F9CB531BC098836B81CC72705D3A8D342613A64FA0EBE09D277D84F7F86C26EC` | `C7884BC93FC05D77FA77C330B2AC2791F51EF52FFD671798FC42DD3127E79A19` |

The exact post-change hashes were captured by the final verification command;
the protected model and dataset hashes above are the integrity gate.

## Recommended next phase

Integrate provenance at the soil-card/OCR extraction boundary. Preserve the
OCR-extracted original value, explicitly supplied unit text, parser result,
and confidence only when the OCR engine provides it. Do not invent units,
confidence, or a physical conversion, and retain unresolved semantics in the
Phase 6 findings.

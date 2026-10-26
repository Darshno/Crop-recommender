# Phase 5 Implementation Plan

This is a plan only. No production code, model, source dataset, or previous
experiment artifact was changed.

## Recommended first change

Implement a small, independently testable **prediction-input semantics status
object** in the shared validation/inference boundary, without changing model
weights or dataset files.

The first slice should:

- preserve raw and transformed rainfall values plus interval/aggregation status;
- preserve soil input origin (`measured`, `ocr`, `level_proxy`, or `soil_type`);
- distinguish invalid values from training-range extrapolation;
- carry `semantics_verified`/`semantics_unresolved` explicitly;
- return a structured result before inference;
- add unit tests for status construction and existing behavior.

The UI wiring can initially display the status and block only when the status is
explicitly unresolved, using a clear experimental-model message. No threshold
should be invented.

## Ordered work items

### P0 — Shared semantics contract

**Goal:** make source, units, sampling/aggregation period, transformation, and
origin explicit for all seven features.

**Acceptance:** a prediction request can state why each value is compatible,
estimated, extrapolated, or unresolved; feature order remains unchanged.

**Tests:** missing metadata, measured values, OCR values, level proxies, and
weather-derived values.

### P0 — Soil-card safety boundary

**Goal:** stop describing quartile mapping as conversion and preserve quantitative
card values/units when available.

**Acceptance:** no unsupported physical formula is introduced; unresolved
compatibility produces an estimate warning or abstention; OCR/manual origin is
visible.

**Tests:** parser fields, OCR ambiguity, unit text, pH parsing, fallback/manual
selection, and no silent numeric discard.

### P0 — Rainfall safety boundary

**Goal:** retain weather provenance and make the 90-day-derived monthly
approximation explicit.

**Acceptance:** raw total, returned-day count, requested interval, transformation,
clamping, and semantic status are available to the UI; unresolved training
period generates an explicit warning or abstention.

**Tests:** date windows, incomplete API responses, malformed/missing values,
clamping, errors, and feature ordering.

### P1 — Reliability policy

**Goal:** separate invalid, extrapolated, estimated, semantically unresolved,
and in-domain states.

**Acceptance:** invalid input never predicts; unresolved semantics cannot appear
as an ordinary confident recommendation; model scores are never described as
success probabilities.

**Tests:** unit tests plus both Streamlit flows or a testable presentation
adapter.

### P1 — Regional evidence transparency

**Goal:** show source, coverage, support, units, and missing-versus-zero status
without producing an unsupported regional suitability score.

**Acceptance:** historical evidence remains in a separate panel; no evidence
means “unavailable,” not zero; pre-cutoff filtering remains strict.

**Tests:** state/season/crop/year filters, zero versus missing, support counts,
regional-only crops, fallback behavior, and source metadata.

### P2 — Threshold evaluation

**Goal:** evaluate candidate abstention policies only after independent,
unit-compatible validation data exists.

**Acceptance:** report selective risk/coverage, class support, calibration,
domain strata, and safety review. Do not select thresholds from the current
random split alone.

## Existing tests inspected

- [tests/test_validation.py](C:/Users/darsh/Downloads/agrisense_phase1/tests/test_validation.py):
  schema, missing/non-numeric values, hard bounds, missing prediction fields,
  extrapolation warnings, training ranges.
- [tests/test_regional_recommender.py](C:/Users/darsh/Downloads/agrisense_phase1/tests/test_regional_recommender.py):
  normalization, strict cutoff, regional-only crops, fallback, invalid values.
- [tests/test_recommend.py](C:/Users/darsh/Downloads/agrisense_phase1/tests/test_recommend.py):
  bundle contract, invalid inference, positive `k`.
- [tests/test_phase3.py](C:/Users/darsh/Downloads/agrisense_phase1/tests/test_phase3.py):
  provenance manifests, units, unresolved input contract, overlap checks.
- [tests/test_metrics.py](C:/Users/darsh/Downloads/agrisense_phase1/tests/test_metrics.py):
  calibration and classification metric validation.

## Protected-file hash verification

SHA-256 hashes were captured before this documentation-only change and checked
again afterward for:

| File | Baseline SHA-256 |
|---|---|
| `app.py` | `B9C003C516C45181FFF3DF9DF8B795ABBC3BD510669A076E2368999E85205539` |
| `app_farmer.py` | `11925BD0D1A20CFE0F3CC48CA045201A642D5E3ABE1A5AB798335B78BA840AF4` |
| `crop_model.py` | `226DB97BBD43E9A3BC8EE18AF2A0A4A3D27B90C5CA3D9B4E69F6347FC228D02E` |
| `regional_recommender.py` | `954627E98D548E3BBE97A74A3F1794A3392FC3E0212145E1F70A3D1CD3138963` |
| `evaluation/validation.py` | `DC07C4078D813A277C26EE0151AC05DFA5312E3658296FAB81A5961ED7B0EF01` |
| `evaluation/metrics.py` | `1764CCFC7F6BB8B58418BA7B6451409FDBA3DCED821C262F237411080F205C4E` |
| `evaluation/phase3.py` | `6C3A8C7C3BD72B01F682A8548753EA986F92A2797CEC81510FEDB75719C1F8A8` |
| `Crop_recommendation.csv` | `54A5A6E5408668E668667EFC50DE2FC867C1B875E0431B4F54DD331B0A109A4E` |
| `crop_model.pkl` | `27A7AA782ED7A3A6E5C6FFC43BFEDC9AAC54D97CFF71316C812C884C49BC16B6` |
| `independent_data.csv` | `2EEB7754870B96D68772471E90BC763E8FCED03D19461158406B45126C20568D` |
| `data/regional/crop_yield.csv` | `AB9BC356B1F8107D490376CEC24450A0E2906322AD25788AB22FB32537AB1F8F` |

## Validation performed

This phase creates documentation only, so no production test run was required.
The existing suite was inspected; the previous Phase 4 run recorded 38 passing
tests. The protected hashes were verified unchanged after document creation.


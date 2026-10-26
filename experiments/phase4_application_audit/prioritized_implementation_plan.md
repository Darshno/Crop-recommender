# Prioritized Phase 4 Implementation Plan

This plan is evidence-based and intentionally does not retrain, tune, change
the pickle, combine datasets, or claim improved accuracy.

## Priority order

| Priority | Work item | Why now | Risk addressed | Effort/regression |
|---|---|---|---|---|
| P0 | Establish a verified train-serving semantics contract for N/P/K, pH, temperature, humidity, and rainfall; fail closed or mark estimates when unresolved | Units and rainfall period are unknown | Highest risk of misleading predictions; user transparency | Medium; central contract may affect both UIs |
| P0 | Preserve raw weather response metadata and remove silent semantic approximation unless the training period is verified | Current 90-day `/3` is only an approximation | Scientific validity and user safety | Medium; weather UX changes |
| P0 | Add explicit reliability states: invalid, out-of-domain, estimated, semantics-unverified, and eligible | Current checks warn but still predict | Misleading extrapolations and overconfidence | Medium |
| P1 | Replace quartile “conversion” wording with a documented proxy workflow, or accept only verified laboratory values | Quartiles are not physical unit conversion | Soil-card misuse | Low/medium; UI copy and input contract |
| P1 | Persist dataset/model hashes, split row IDs, environment lock, and benchmark manifest | Current seed is reproducible but split provenance is not persisted | Auditability and regression control | Low |
| P1 | Expand evaluation to independent, unit-compatible field data with farm/time/district holdouts | Random split cannot establish real-world validity | Scientific validity | High; requires data collection |
| P1 | Add domain-aware abstention/coverage evaluation and thresholds selected only from independent validation | No safe threshold exists yet | User safety | Medium/high |
| P2 | Add end-to-end UI tests for both Streamlit entry points, weather failures, OCR ambiguity, feature order, and regional fallback | Existing tests are module-level only | Regression risk | Medium |
| P2 | Add regional provenance/coverage UI: source, years, observation count, missing-vs-zero status, and unit disclaimer | Regional values are easy to overread | Transparency | Medium |
| P2 | Add local explanation support only as model-behavior explanation, with caveats | No runtime explanation exists | Prevents feature importance being read as agronomic proof | Medium |

## Acceptance evidence

- A versioned contract states each feature's unit, sampling basis, period, and
  permitted transformations, with unresolved fields blocking production claims.
- Weather tests prove the displayed value and model value have the same declared
  period and preserve source coverage.
- Soil-card tests distinguish parsed values, units, confidence, and proxy
  category; no unsupported conversion formula is introduced.
- Reliability tests cover malformed, non-finite, physically implausible,
  out-of-training-range, and semantically unresolved inputs.
- Evaluation reports include immutable hashes, row-level split provenance,
  independent validation protocol, calibration/coverage results, and explicit
  limitations.
- Regional tests prove strict pre-cutoff filtering, state/crop/season matching,
  missing-vs-zero handling, and separation from classifier classes.

## Guardrails

- Do not combine independent datasets without provenance, measurement
  compatibility, target compatibility, and a documented protocol.
- Do not describe model scores as crop success probabilities or yield.
- Do not describe global feature importance or SHAP contributions as proof of
  agronomic suitability.
- Do not claim accuracy or real-world improvement until independent validation
  is completed.


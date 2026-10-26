# Phase 5 Prediction Reliability Design

## Design principles

- A model score ranks model classes; it is not the probability of successful
  cultivation, yield, profitability, or agronomic suitability.
- Invalidity, extrapolation, semantic uncertainty, and missing regional
  evidence are separate states.
- No threshold is proposed as a fact. Candidate thresholds must be selected
  and evaluated on appropriate validation data, not invented from the current
  2,200-row benchmark.

## Proposed state model

| State | Definition | User behavior | Current evidence |
|---|---|---|---|
| Invalid input | Missing, non-numeric, non-finite, or outside current broad validation bounds | Reject; do not call model | `validate_input_values()` in [evaluation/validation.py](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/validation.py) |
| In-domain input | Valid and within stored training min/max, with verified units/period/method | Allow ranking with clear model-score disclaimer | Training ranges are stored in the bundle |
| Training extrapolation | Valid physical-range value below training minimum or above maximum, or unusual beyond q01/q99 | Warn prominently; do not silently present as normal | Current code warns but still predicts |
| Semantics unverified | Unit, sampling basis, OCR provenance, or rainfall aggregation is not compatible/known | Prefer abstention; if exploratory mode exists, label result as estimate and require acknowledgement | Current code does not represent this state |
| Estimated input | Soil type or Low/Medium/High proxy rather than a verified measurement | Label all output as estimate; safer default is no production recommendation | Farmer UI has partial warning only |
| Regional evidence missing/sparse | No matching records, no pre-cutoff records, invalid records, or support below configured rule | Show classifier result separately; state “no historical evidence,” not “low suitability” | Current regional fallback preserves base ordering |

## Reliability controls

### Confirmed by source code or data

- Existing validation handles malformed values and broad plausibility bounds.
- Existing training ranges support extrapolation warnings.
- Existing UI text says model scores are not crop-success probabilities.
- Regional fallback does not alter the base ordering when evidence is absent.

### Inferred

- Warning-only extrapolation is insufficient for a farmer-facing decision path
  when the input semantics are unresolved.
- A typed prediction context is needed so the UI cannot lose whether a value was
  measured, OCR-derived, proxy-mapped, clamped, or semantically unresolved.

### Unresolved

- Safe abstention threshold and acceptable coverage/risk trade-off.
- Whether a particular score margin, entropy, or probability threshold is safe.
- Domain-specific physical ranges and measurement compatibility.

## Threshold evaluation plan (no thresholds invented)

Candidate policies may be compared only after compatible validation data exists:

1. Define candidate abstention rules using input-domain status, semantic status,
   score margin/entropy, and calibration diagnostics.
2. Evaluate coverage, top-1/top-k error, macro-F1, calibration, worst-class
   performance, and selective risk on independent field data.
3. Stratify by measured versus estimated input, in-range versus extrapolated
   input, geography, season, and crop class.
4. Select a policy with agronomist/user-safety review; document false-accept and
   false-abstain costs.
5. Revalidate after any data, model, or input-contract change.

## Proposed UI language

- “Model score” rather than “success probability.”
- “Input outside observed training range” rather than “bad soil.”
- “Rainfall period compatibility is unverified.”
- “Historical evidence unavailable” rather than “crop is unsuitable.”
- “This is an experimental model ranking and not agronomic advice.”

## Tests for subsequent implementation

- Invalid input rejects inference.
- Each extrapolation category produces a stable, feature-specific warning.
- Semantic-unverified inputs cannot accidentally enter the normal-success UI.
- Estimated and measured soil paths remain distinguishable.
- Model-score wording is present and crop-success wording is absent.
- Regional missing/sparse evidence leaves the classifier output separate.
- Feature order remains exactly `N, P, K, temperature, humidity, ph, rainfall`.


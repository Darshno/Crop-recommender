# Prediction Reliability Audit

## Existing controls

### Confirmed from source code or data

- [`evaluation/validation.py`](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/validation.py)
  rejects missing, non-numeric, non-finite, and broad-bound violations.
- It warns when values are outside training min/max and flags unusual values
  outside training 1st/99th percentiles. These ranges are stored from the
  training split in the model bundle.
- `crop_model.py` produces calibration bins, ECE, Brier score, a reliability
  diagram, and jitter sensitivity. These are diagnostics, not guarantees.
- Both UIs state that scores are not probabilities of crop success or expected
  yield. The farmer UI marks soil profiles as estimates.
- Regional fallback preserves the base ordering when evidence is unavailable.

## Gaps and interpretation

### Confirmed from source code or data

- The hard bounds are broad plausibility bounds, not crop- or
  measurement-method-specific validity ranges.
- Out-of-training-range values generate warnings but do not prevent
  prediction. Weather values are clamped before validation, which can hide the
  original out-of-range observation.
- No abstention threshold, conformal set, selective-risk report, or
  per-class uncertainty interval is implemented.
- No runtime SHAP explanation is implemented. Global feature importance in
  training output is not evidence that a feature is agronomically causal or
  suitable.

### Inferred and requiring verification

- Calibrated test-set probabilities may be better aligned with held-out
  classification frequency, but calibration quality can fail under distribution
  shift and does not mean real-world success probability.
- A warning-only extrapolation policy may be unsafe for farmer-facing use when
  units or rainfall semantics are unresolved.

### Unresolved

- There is no independent evidence for a safe confidence threshold or
  abstention policy.
- Physical plausibility limits for each measured feature cannot be finalized
  without the measurement units and sampling methods.

## Recommended reliability behavior

1. Parse and validate all inputs before clamping; retain raw and transformed
   values and show transformations.
2. Separate hard invalidity from out-of-training-domain warnings.
3. Add a “semantics verified / estimate / unresolved” status and abstain by
   default when required units or rainfall period are unresolved.
4. Report top-k ranking plus calibrated model diagnostics only as model
   confidence, never agronomic success.
5. Add perturbation/domain checks and require independent field validation
   before any suitability claim.
6. If explanations are added, describe them as local model behavior
   (e.g. contribution/sensitivity), not proof of agronomic suitability.


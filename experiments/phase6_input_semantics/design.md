# Phase 6 Shared Prediction-Input Semantics Layer Design

## Scope

Phase 6 adds a structured metadata and validation context before inference. It
does not modify model weights, training data, dataset files, feature order, or
the meaning of any existing numeric input.

The implementation is in
[evaluation/input_semantics.py](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/input_semantics.py).

## Contract

The canonical feature order remains exactly:

```text
N, P, K, temperature, humidity, ph, rainfall
```

Each `InputField` records:

- feature name;
- numeric value when the existing validator accepts it;
- origin (`manual`, `ocr`, `categorical_level_proxy`, `weather_api`, or another
  caller-supplied label);
- original value;
- ordered transformation history;
- unit and unit-status metadata;
- measurement-method status;
- OCR confidence only when supplied by the caller;
- source metadata.

No unit, confidence, compatibility, or conversion is inferred by the layer.

## Findings and state precedence

Findings are a tuple and are intentionally non-exclusive:

- `INVALID_INPUT`
- `OUTSIDE_TRAINING_RANGE`
- `UNRESOLVED_SEMANTICS`
- `OCR_UNCERTAINTY`
- `MISSING_SOURCE_METADATA`

Primary state is derived narrowly:

1. Any invalid finding -> `INVALID`.
2. Any unresolved semantic, OCR uncertainty, or required-metadata finding ->
   `SEMANTICS_UNRESOLVED`.
3. Otherwise any training-range finding -> `EXTRAPOLATED`.
4. Otherwise -> `VALIDATED`.

`VALIDATED` means only that the implemented numeric validation checks passed and
no explicitly supplied unresolved finding was recorded. It does not mean that
units are scientifically verified, model probabilities are calibrated for
agriculture, or a recommendation is agronomically suitable.

## Rainfall provenance

`RainfallProvenance` records source, original value/unit, start/end dates or
interval, aggregation method, transformed value, transformation history, and
whether compatibility with the training feature has been established.

The object records the existing `/3` transformation when a caller supplies it;
it does not implement or endorse that transformation. Compatibility must be
explicitly supplied as `True` to avoid silently claiming it.

## Validation reuse

`build_prediction_input()` delegates numeric parsing, finite checks, hard bounds,
and training-range diagnostics to the existing
[evaluation/validation.py](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/validation.py).
This avoids a second numeric-validation implementation.

## Serialization

`PredictionInputSemantics.to_dict()` and `from_dict()` support explicit
serialization/deserialization for audit logs or UI handoff. The layer does not
store precise user location or other unnecessary personal information; source
metadata is caller-controlled.

## Integration boundary

`crop_model.recommend_with_input_semantics()` loads the unchanged bundle,
builds the semantics context, validates the numeric values, performs inference
in the existing feature order, and returns both `(recommendations, semantics)`.
The existing `recommend()` function delegates to it and keeps its original
`list[(crop, score)]` return contract.

The Streamlit pages still call `predict_proba()` directly. They do not yet have
structured metadata objects for all UI branches, especially weather and OCR.
Wiring them directly would risk silently labeling legacy inputs with metadata
that is not actually available. This integration gap is documented rather than
filled with invented units or confidence values.


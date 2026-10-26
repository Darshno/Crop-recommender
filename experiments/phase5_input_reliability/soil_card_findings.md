# Phase 5 Soil-Card Input Semantics Findings

## Scope and status vocabulary

This is a read-only investigation. Findings are classified as **Confirmed by
source code or data**, **Inferred**, or **Unresolved**. No production source,
dataset, model pickle, or prior experiment artifact was changed.

## End-to-end trace

### `app.py`

| Source field | Parsed value | Unit | Current transformation | Expected model input | Compatibility evidence | Unresolved assumptions |
|---|---|---|---|---|---|---|
| Low/Medium/High nitrogen selector | `n_lv` | Not documented | `level_value("N", level)` selects training CSV 25th/50th/75th percentile; fallback `(21, 37, 84)` if CSV load fails | `N` | `SOIL_Q`, `LEVEL_IDX`, and `level_value()` in [app.py](C:/Users/darsh/Downloads/agrisense_phase1/app.py) | Training unit, extractability, soil depth, lab method, and Soil Health Card scale are unknown |
| Low/Medium/High phosphorus selector | `p_lv` | Not documented | Same quartile lookup; fallback `(28, 51, 68)` | `P` | Same source | Same unresolved semantics |
| Low/Medium/High potassium selector | `k_lv` | Not documented | Same quartile lookup; fallback `(20, 32, 49)` | `K` | Same source | Same unresolved semantics |
| pH category selector | `ph_ch` | pH is dimensionless in principle | Fixed representative value: 5.5, 6.8, or 7.8 | `ph` | `PH_VALUES` in [app.py](C:/Users/darsh/Downloads/agrisense_phase1/app.py) | Sampling method/depth and whether representative values reflect training data |
| Named soil type | `soil` | None | Hard-coded soil profile maps to levels/categories, then same quartile and pH mapping | `N`, `P`, `K`, `ph` | `SOIL_TYPES` in [app.py](C:/Users/darsh/Downloads/agrisense_phase1/app.py) | Soil color/type is not a measurement and has no site-specific evidence |
| Exact slider values | N/P/K/pH numbers | UI labels only; training units not confirmed | Passed directly after numeric validation | `N`, `P`, `K`, `ph` | Exact branch in [app.py](C:/Users/darsh/Downloads/agrisense_phase1/app.py) | Slider ranges are product choices, not evidence of training-unit compatibility |

### `app_farmer.py`

| Source field | Parsed value | Unit | Current transformation | Expected model input | Compatibility evidence | Unresolved assumptions |
|---|---|---|---|---|---|---|
| OCR nitrogen line | First numeric value, or Low/Medium/High rating | `CARD_LIMITS` comment says typical kg/ha, but source confirmation is absent | `parse_lines()` recognizes “nitrogen”; numeric values are classified by `level_of()` and later reduced to a level; `vals()` replaces level with training quartile | `N` | `parse_lines()`, `CARD_LIMITS`, `level_of()`, and `vals()` in [app_farmer.py](C:/Users/darsh/Downloads/agrisense_phase1/app_farmer.py) | Card state/version, exact unit, extractable basis, depth, OCR confidence, and threshold authority |
| OCR phosphorus line | First numeric value, or rating | Same unresolved card-unit issue | Same level conversion; `P2O5` text is recognized | `P` | Same source | No demonstrated compatibility with training `P` |
| OCR potassium line | First numeric value, or rating | Same unresolved card-unit issue | Same level conversion; `K2O` text is recognized | `K` | Same source | No demonstrated compatibility with training `K` |
| OCR pH line | First numeric pH in [3,10] | Dimensionless in principle | `ph_level()` maps `<6`, `6–7.5`, `>7.5` to 5.5/6.8/7.8 | `ph` | `parse_lines()` and `ph_level()` | Sampling method/depth and representative-value validity |
| Manual card selectors | Level indices | Not applicable | `vals()` maps levels to training quartiles and fixed pH representatives | `N/P/K/ph` | `finish_card()` in [app_farmer.py](C:/Users/darsh/Downloads/agrisense_phase1/app_farmer.py) | Same unresolved semantics |
| Exact card sliders | Numeric N/P/K/pH | UI labels only | Direct values when `use_exact` is selected | `N/P/K/ph` | `finish_card()` | Unit/method still not established |

## Findings

### Confirmed by source code or data

- Both UIs use the shared feature order from the loaded model bundle.
- Low/Medium/High N/P/K is mapped to empirical training-data quartiles,
  not converted using a documented physical formula.
- `app_farmer.py` uses `CARD_LIMITS = {"N": (280, 560), "P": (10, 25), "K": (110, 280)}`
  and labels the comment as typical Soil Health Card ratings, with “check your
  state chart”; this is not a repository-level unit specification.
- The farmer UI describes soil profiles as approximate/not lab values and
  advises soil testing.
- `validate_input_values()` validates numeric/finite/broad-bound values and
  warns about training-range extrapolation, but it does not validate units,
  sampling method, OCR confidence, or semantic provenance.

### Inferred

- The application intends quartiles to preserve a rough ordinal distinction
  (low/medium/high), not to claim measured equivalence. That intent should be
  made explicit in the UI and prediction context.
- OCR values are more informative than a level-only card, but the current
  level reduction discards that quantitative information before prediction.

### Unresolved

- The original N/P/K training units and laboratory methods.
- Soil Health Card units, extractable nutrient basis, sampling depth, state
  thresholds, and whether card values are directly comparable.
- Whether pH category representatives are appropriate for the model's training
  distribution.
- OCR confidence and whether the selected value came from OCR or manual input.

## Safe handling proposal

Until compatibility is verified:

1. Keep quantitative card values and their declared units as provenance.
2. Do not call quartile mapping a conversion. Label it “training-distribution
   proxy” and identify the prediction as estimated.
3. Offer a quantitative-input path only when the user can provide the value,
   unit, sampling basis, and method needed by the verified contract.
4. If required semantics are absent, show the limitation and abstain from a
   production recommendation rather than inventing a conversion.
5. Do not introduce thresholds or formulas until supported by authoritative
   card documentation and compatibility evidence.


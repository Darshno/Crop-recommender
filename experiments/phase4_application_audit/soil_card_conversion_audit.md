# Soil-Card Conversion Audit

## Conversion inventory

| Source field | Parsed value | Assumed unit | Transformation | Expected model input | Evidence | Uncertainty |
|---|---|---|---|---|---|---|
| `app.py` Low/Medium/High N | Selected level | None documented | Training CSV 25th/50th/75th percentile via `SOIL_Q`; fallback constants if CSV load fails | `N` | `level_value()` and `SOIL_Q` in [`app.py`](C:/Users/darsh/Downloads/agrisense_phase1/app.py) | Training units and Soil Health Card scale are not established; not a physical conversion |
| `app.py` Low/Medium/High P | Selected level | None documented | Same quartile lookup | `P` | `level_value()` | Same unresolved unit/scale issue |
| `app.py` Low/Medium/High K | Selected level | None documented | Same quartile lookup | `K` | `level_value()` | Same unresolved unit/scale issue |
| `app.py` pH category | Category | pH is dimensionless in principle, but source method is unknown | Fixed 5.5, 6.8, or 7.8 | `ph` | `PH_VALUES` | Category boundaries and representative values are product assumptions, not source calibration |
| `app_farmer.py` card OCR N/P/K | OCR number or Low/Medium/High text | `CARD_LIMITS` comments say typical kg/ha | Numeric values are classified into low/medium/high, then replaced by training quartile value in `vals()` | `N/P/K` | `parse_lines()`, `level_of()`, `vals()` in [`app_farmer.py`](C:/Users/darsh/Downloads/agrisense_phase1/app_farmer.py) | The comment says “check your state chart”; OCR number units, extraction confidence, depth, and state-specific thresholds are unresolved |
| `app_farmer.py` card OCR pH | OCR pH number | Dimensionless in principle | `ph_level()` maps `<6`, `6–7.5`, `>7.5` to fixed 5.5/6.8/7.8 | `ph` | `parse_lines()` and `PH_VALUES` | Sampling method/depth and representative-value validity are unresolved |
| Soil-type choice in both apps | Named soil type | None | Hard-coded approximate profile to Low/Medium/High and pH category, then same quartile mapping | `N/P/K/ph` | `SOIL_TYPES` / `SOIL_PROFILE` | Not a soil test; geographic and laboratory variability is not represented |

### Confirmed from source code or data

- Exact-value sliders bypass the Low/Medium/High mapping but still do not
  document that their ranges/units match the training data.
- OCR is optional and can fail; ambiguous or missing fields fall back to user
  radio selections. OCR exceptions are not surfaced with a detailed reason.
- `app_farmer.py` labels profiles as approximate/not lab values and warns that
  estimates should be confirmed by soil testing.

### Inferred and requiring verification

- The intended Soil Health Card numeric scale may be kg/ha, but the repository
  does not establish that the training N/P/K columns use the same extractable
  nutrient basis, depth, laboratory method, or unit.
- Quantile mapping preserves relative rank categories only if the card scale and
  training distribution are compatible; that compatibility has not been shown.

### Unresolved

- No defensible unit conversion formula can be supplied from repository
  evidence. The implementation must not replace the proxy with an invented
  formula.
- OCR confidence, card version/state, sampling depth, lab method, and timestamp
  are not retained as model-input metadata.

## Implementation implications

The safest next step is a fail-closed input contract: accept a documented
laboratory measurement only when its unit/method is verified against the
training semantics; otherwise label the result as an estimate or abstain.
Do not claim that quartile mapping is a physical conversion or that a soil-type
profile is a measured soil observation.


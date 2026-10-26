# Recommended data strategy

## Recommendation: D — keep the current model and acquire better-labelled data

Do not combine the two local CSVs. Do not retrain or tune the existing model
from the current evidence.

### Why

- Local schema compatibility is only syntactic. Units and measurement methods
  are unknown for every feature.
- The two datasets have materially different shared-class feature-label
  relationships.
- The independent file has no provenance or label-definition evidence.
- The previous 416-row shared-class evaluation covers only cotton, maize, and
  rice, and the experimental model scored 0/416.
- No authoritative source was found that can be shown to be the origin of both
  local files or that directly supplies all seven compatible fields and the
  same target.

### Evidence required before any combined model

1. Immutable source manifests with URL/version/checksum and license.
2. Unit, soil-depth, extraction, instrument, and sampling metadata.
3. Weather source, coordinates, dates, timezone, and aggregation windows.
4. An explicit target definition and label-verification process.
5. Field/farm/source/time identifiers for grouped splitting.
6. A prespecified final evaluation set covering every claimed class.
7. Expert review of feature and label harmonization.

### Acceptable future alternatives

- **B — separate model for the independent source:** only after its labels,
  units, and provenance are established. It must have its own grouped
  validation and final test set.
- **C — compatible subset:** only after a documented subset passes unit,
  method, geography, time, and target checks. Shared column names alone are
  insufficient.
- **A — combine after harmonization:** not currently justified; it becomes
  possible only after all evidence requirements are satisfied.

### What not to do

- Do not remap labels based on similar names or predictions.
- Do not convert units using quantiles or model ranges.
- Do not clip outliers to make distributions match.
- Do not tune against the 416-row evaluation set.
- Do not claim improved generalization from the existing results.

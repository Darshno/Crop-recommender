# Independent-data v1 forensic audit

## Conclusion

The 0% experimental result is not explained by feature order or a missing label-encoder class. The audit confirms that the experimental model predicts labels different from the three eligible ground-truth labels; distribution shift and unverified units/label provenance remain plausible causes requiring source verification.

## Key evidence

- Eligible evaluation rows: **416**.
- Baseline accuracy: **0.9615%**.
- Experimental accuracy: **0.0000%**.
- Feature order match: **True**.
- All seven independent labels in experimental encoder: **True**.
- Mean standardized nearest-training distance: **1.2485**.

## Confirmed problems

1. The experimental model fails on every eligible row; see `evaluation_predictions.csv` and `prediction_distribution.csv`.
2. The direct evaluation covers only cotton, maize, and rice, not the full independent label set.

## Potential problems requiring source verification

1. Feature units and measurement procedures are undocumented.
2. Independent feature distributions may be shifted beyond original observed ranges.
3. Independent crop labels may not have the same provenance or operational definition as original labels.

No generalization improvement is supported. Existing source files and v1 artifacts were not modified.

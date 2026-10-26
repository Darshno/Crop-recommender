# Recommended leakage-free evaluation protocol

1. Obtain a provenance and unit manifest for every source. Reject evaluation
   claims until feature semantics and label definitions are documented.
2. Freeze the baseline model and encoder. Do not fit preprocessing or choose
   thresholds using final test rows.
3. Attach immutable source, farm/field/site, geography, season, date, and
   provenance identifiers to every row. Keep duplicate and related observations
   in the same split.
4. Define the target explicitly and require reliable ground-truth labels,
   preferably agronomist-reviewed labels plus measured outcomes where the claim
   concerns suitability or success.
5. Build a development set from verified training sources. Use grouped
   train/validation splits by farm/source/time; use a final untouched grouped
   test set.
6. For the 22-class baseline, evaluate only on final-test rows whose labels are
   in the frozen baseline encoder. Report all supported classes, class support,
   absent classes, balanced accuracy, macro-F1, weighted-F1, per-class metrics,
   and confusion matrices.
7. For the expanded model, define the full claimed class set and ensure every
   claimed evaluation class is represented in the final test set. Evaluate all
   26 classes separately; do not infer performance for absent classes.
8. Keep independent-only classes separate from the baseline comparison. A
   baseline cannot be scored on labels it cannot emit.
9. Report duplicate removal, conflict handling, source counts, group counts,
   class coverage, feature-range coverage, and unit/provenance limitations.
10. Repeat the final evaluation only on a new, untouched external collection
    after the protocol is frozen. Do not tune against the final test set.

This protocol is a recommendation only. No retraining or tuning was performed
in Phase 2, and no improved generalization is established.

# Phase 2 independent-data investigation

## Scope and constraints

This is a read-only descriptive audit. No model was trained, tuned, replaced,
or loaded for prediction during this phase. Source CSVs and all v1 artifacts
were left untouched.

## Confirmed findings

- Original dataset: **2,200** rows and 22 classes.
- Independent dataset: **1,000** rows and 7 classes.
- Shared classes: **cotton, maize, rice**.
- Exact cross-source feature-vector matches: **0**.
- Near feature-similarity label disagreements at the heuristic threshold
  (standardized Euclidean distance <= 1.0): **36**.
- Per-class statistics and quantiles are in `shared_class_statistics.csv`.
- Per-feature/crop range overlap and mean/median shifts are in
  `feature_label_analysis.csv`.

Different distributions do not automatically mean either file is incorrect.
They do show that source populations or data-generation processes differ and
must be explained before combining labels for retraining.

## Feature-label relationships

The shared-class analysis compares all seven features for cotton, maize, and
rice. `feature_label_relationships.png` and the `shared_class_*.png` files
show the distributions. Range overlap is descriptive only: overlapping
feature values with different crop labels can reflect genuine ambiguity,
different environments, label definitions, or incompatible units.

`conflicting_label_groups.csv` contains near feature-similarity pairs with
different labels. These are **not declared ground-truth contradictions**:
the distance is a heuristic and no unit/provenance evidence exists. Exact
cross-source matches are in `exact_cross_source_feature_matches.csv`.

## Nearest-neighbor analysis

`nearest_neighbor_summary.csv` summarizes every incorrectly predicted
independent evaluation row, including the nearest training source, label, and
standardized distance. Distances use StandardScaler fit on original feature
columns followed by Euclidean distance. This makes feature magnitudes
comparable for a descriptive diagnostic, but it does not establish that the
features have compatible units or agronomic meaning.

## Dataset compatibility

Repository documentation describes the original file as the commonly used
2,200-row, 22-class crop recommendation dataset and notes unresolved
provenance, synthetic/augmented-data concerns, and undocumented feature
semantics. The independent CSV has no manifest, source reference, collection
method, geography, observation period, unit specification, farm IDs, dates,
or agronomist verification. These are confirmed metadata gaps, not evidence
that the independent records are invalid.

## Evaluation limitation

The prior 416-row shared-class evaluation covers only cotton, maize, and rice.
It cannot establish performance for all 26 classes in the experimental model,
and it cannot support a comparison for independent-only classes that the
baseline encoder does not support.

## Potential compatibility problems

1. Feature distributions differ substantially within shared classes.
2. The independent source may use different units, sampling procedures, or
   time aggregation windows.
3. Labels may represent different targets (for example, observed crop,
   recommended crop, or another assignment) even though both columns are
   named `label`.
4. Near feature-similarity pairs frequently have different crop labels; this
   requires source verification rather than automatic conflict removal.

No model-performance improvement is claimed.

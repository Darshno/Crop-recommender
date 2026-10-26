# Baseline Evaluation Audit

## Dataset and reproducibility

### Confirmed from source code or data

- `Crop_recommendation.csv` has 2,200 rows, eight columns
  (`N,P,K,temperature,humidity,ph,rainfall,label`), and 22 crop classes
  (100 rows per class, as documented in [`README.md`](C:/Users/darsh/Downloads/agrisense_phase1/README.md)).
- `crop_model.py` fixes `RANDOM_STATE=42`, `TEST_SIZE=0.20`,
  `CV_SPLITS=5`, and `CALIBRATION_SPLITS=3`.
- `train_test_split(..., stratify=y, shuffle=True, random_state=42)` creates
  the held-out split. Candidate selection uses only the training portion with
  shuffled five-fold `StratifiedKFold`, selecting mean macro-F1 with balanced
  accuracy and accuracy tie-breaks.
- The selected estimator is calibrated with sigmoid calibration using
  three-fold CV inside the training portion. The test rows are not used to fit
  or select the model.
- The saved bundle records feature order, selected model name, random state,
  split fraction, calibration method, class list, row counts, dataset filename,
  and sklearn version. `environment.json` records Python/platform and package
  versions when the training script is run.
- Existing benchmark generation calculates accuracy, balanced accuracy,
  macro precision/recall/F1, top-3 accuracy, log loss, multiclass Brier score,
  top-label ECE, per-class report, confusion matrix, calibration bins,
  reliability diagram, and a synthetic jitter sensitivity diagnostic.
- [`evaluation/dataset_audit.py`](C:/Users/darsh/Downloads/agrisense_phase1/evaluation/dataset_audit.py)
  checks exact/rounded train-test feature overlap and nearest standardized
  neighbors; it fits its diagnostic scaler on training rows only.

### Inferred and requiring verification

- The process is reproducible at the code/configuration level if the exact CSV
  bytes, Python/package versions, and model-training command are preserved.
  Reproducing the current pickle exactly also depends on the original runtime
  and serialized estimator state, which should be verified from artifact
  metadata/hash.
- Random stratification is appropriate for a grouped-by-label file, but it does
  not represent held-out farms, districts, seasons, or time periods.

### Unresolved

- There is no saved train/test row-index manifest in the visible benchmark
  artifacts. The split can be regenerated from the seed, but auditability
  would be stronger with persisted source row IDs and hashes.
- No farm, district, coordinate, date, season, soil sampling, variety, or
  measured outcome fields are present. Real-world agricultural validity cannot
  be established from this dataset alone.

## Leakage and preprocessing

### Confirmed

- The production classifier uses the seven raw numeric columns; no fitted
  preprocessing or feature-selection transform is used for the Random Forest
  path.
- Candidate pipelines that need scaling keep scaling inside a sklearn
  `Pipeline`, and CV is performed on the training portion.
- The application computes soil quartiles from the full local CSV at UI import
  time, but this is input mapping, not a training-time feature transform.

### Inferred and requiring verification

- If quartile-derived UI values are treated as measurements, they can create
  semantic leakage from the benchmark distribution into serving inputs even
  though they do not leak test labels. This is a validity/measurement issue,
  not evidence that the benchmark split itself is numerically leaked.

## Required additional evaluation

1. Persist immutable dataset/model/split hashes and source row IDs.
2. Obtain genuinely independent, labelled field data with documented units,
   sampling protocol, geography, dates/seasons, and label/outcome provenance.
3. Evaluate farm-, district-, season-, and time-held-out protocols where IDs
   exist; never combine unverified datasets for retraining.
4. Report per-class support, macro metrics, confusion, top-k, log loss, Brier,
   calibration, and abstention/coverage by in-domain versus extrapolation
   inputs.
5. Test unit/aggregation compatibility before any external score is interpreted.
6. Add UI/inference contract tests for feature ordering, malformed values,
   training-range warnings, weather failures, OCR ambiguity, and regional
   fallback.


# Independent-data v1 experiment

This is an opt-in experiment. It does not modify the application, the frozen
`crop_model.pkl`, `Crop_recommendation.csv`, or `independent_data.csv`.

From the project root, run:

```powershell
.venv\Scripts\python.exe experiments\independent_data_v1\experimental_pipeline.py
```

The pipeline normalizes harmless header/label whitespace and case differences
in memory, rejects missing/non-numeric/non-finite/out-of-bound values, detects
exact duplicates and feature-only label conflicts, and writes all outputs in
this directory. Conflicting feature vectors are exported and excluded.

Evaluation rows are independent rows whose normalized feature vectors do not
occur in the original data or the training data. For a direct baseline
comparison, only labels known by the frozen baseline encoder are eligible.
The output explicitly reports excluded labels, class coverage, and whether the
sample is sufficient for a cautious generalization statement. The files do not
contain unit or provenance metadata, so unit compatibility remains unverified;
the pipeline does not clip, convert, or invent values.

Important outputs include `data_audit.json`, `duplicate_summary.json`,
`cleaned_combined_dataset.csv`, `training_report.json`,
`per_class_metrics.csv`, `confusion_matrix.csv`,
`baseline_confusion_matrix.csv`, and
`crop_model_experimental_v1.pkl`.

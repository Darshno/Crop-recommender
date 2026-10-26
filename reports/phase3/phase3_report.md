# Phase 3 — Real-world validity and train-serving consistency

## Current verdict

**Not yet validated for real-farm accuracy.** Phase 3 adds a fail-closed framework for independent labelled evaluation and records the current app/model semantic mismatches. It does not fabricate external validation results: no independent field dataset was included with this project.

## Input semantics audit

The current farmer app obtains weather for a district/town and computes:
- temperature: mean of daily mean temperature over the most recent 90-day window;
- humidity: mean of daily mean relative humidity over the same window;
- rainfall: total precipitation over 90 days divided by three, an approximate monthly amount;
- N/P/K: soil-card ratings are converted to low/medium/high indices and then mapped to quartiles from the model CSV;
- pH: a coarse value selected from three fixed levels.

The training CSV only contains N, P, K, temperature, humidity, pH, rainfall and a crop label. It does not document the original units, sampling methods, time windows, source row IDs or provenance per row. Therefore, there is no evidence in this repository that live app inputs are semantically equivalent to the training features. In particular, a monthly rainfall estimate must not be assumed equivalent to a growing-season rainfall total. Likewise, soil-card units must not be converted by using training-data quartiles as if they were a physical unit conversion.

**Action:** obtain authoritative documentation for the exact training CSV and align app inputs with it. If the definitions cannot be recovered, label the output as an experimental model score and do not use it as a validated agronomic recommendation.

## Independent evaluation

`evaluate_external.py` evaluates the existing frozen model against a separately supplied CSV. It does not fit or tune the model. The CSV must contain all seven model features and a `label` column. The accompanying JSON manifest must document the data source, collection method, geography, observation period, label definition, units for every feature, and explicitly confirm independence from training data.

Example:

```powershell
python evaluate_external.py --data path\to\independent_data.csv --manifest path\to\external_manifest.json
```

Start from `external_manifest.template.json`, replace every placeholder with true information, and do not assert independence unless you have checked it. The tool refuses incomplete provenance, unknown crop labels, invalid input columns, and a byte-identical copy of the configured training CSV.

Outputs are written to `reports/phase3/external_evaluation/`:
- `external_evaluation.json`
- `per_crop_classification_report.csv`
- `confusion_matrix.csv`

Metrics include accuracy, balanced accuracy, macro precision/recall/F1, top-3 accuracy, log loss, multiclass Brier score and top-label ECE. The report also records missing crop classes and whether farm IDs, row-level dates, expert-verified labels and measured outcomes are available.

## Important interpretation

An independent dataset may still be unrepresentative or weakly labelled. High scores against a record of which crop was planted do not necessarily prove that the crop was the best choice, yielded well, or was profitable. Stronger evidence needs traceable field/site/season identifiers, documented units and sampling protocols, agronomist-reviewed labels, and preferably measured harvest outcomes. Where multiple rows come from the same field, use field/source/time-grouped evaluation rather than randomly splitting rows.

## Tests

Run `python -m pytest tests/ -v`. Phase 3 tests validate the provenance gate and the input semantics contract. No external-data performance is claimed until a real independent dataset is supplied.

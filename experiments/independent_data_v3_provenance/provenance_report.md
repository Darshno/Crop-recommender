# Phase 3 — Data provenance and source validation

## Decision

**Do not combine `Crop_recommendation.csv` and `independent_data.csv`.**
The available evidence does not establish compatible units, measurement
methods, geography, time coverage, or label definitions. No model was trained
or tuned in this phase.

The recommended strategy is **D: keep the current model frozen and acquire
better-labelled, provenance-complete data**. A separate model for the
independent source is only a future option after its target and measurement
semantics are documented.

## Verified repository evidence

### Original dataset

Verified from [REPORT.md](../../REPORT.md), [README.md](../../README.md),
`reports/phase2/dataset_audit.md`, and the local CSV:

- The local file has 2,200 rows, 22 classes, and seven numeric features.
- Project documentation calls it the “Kaggle Crop Recommendation Dataset” and
  credits “Atharva Ingle”.
- The documented Kaggle location is:
  <https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>.
- The repository documentation says the commonly used dataset is clean and
  partly synthetic/augmented, but the local CSV has no row-level provenance.
- The CSV does not define units, sampling depth, weather aggregation windows,
  geography, dates, label assignment, or measured agricultural outcomes.

The Kaggle attribution is verified as a repository claim, not independent
confirmation of the local file's exact lineage.

### Independent dataset

Verified from the local file and prior audits:

- The local file has 1,000 rows and seven classes.
- It has the same column names as the original file.
- It has no citation, URL, version, collection method, geography, time period,
  units, field identifiers, label protocol, or outcome fields.
- Its provenance cannot be established from this repository.

The independent file must therefore be treated as **unknown provenance**, not
as verified field data.

### Feature and label semantics

The seven column names are present and numeric, but names alone do not verify
semantics:

- `N`, `P`, `K`: units, soil basis, depth, extraction method, and laboratory
  method are undocumented.
- `temperature`, `humidity`: aggregation period, location, sensor/reanalysis
  source, and timezone are undocumented.
- `ph`: sampling depth and measurement method are undocumented.
- `rainfall`: temporal aggregation is undocumented.
- `label`: both files use the same column name, but neither file says whether
  it means planted crop, recommended crop, optimal crop, expert label, or
  observed outcome.

The application documentation explicitly says that its soil-card values are
mapped through training-data quartiles and that 90-day precipitation is
divided by three to create an approximate monthly value. This is a
train-serving semantic risk, not evidence that the CSV rainfall feature is
monthly.

## External source investigation

External research identified authoritative component sources, but no verified
single dataset containing all seven compatible fields and the same crop-label
target:

1. **Government of India Soil Health Card / Open Government Data Platform**
   - <https://www.data.gov.in/keywords/Soil%20Health%20Card>
   - <https://soilhealth.dac.gov.in/>
   - Relevant for field soil measurements and soil-health recommendations.
   - Exact downloadable resource, units, sampling method, and crop-label
     semantics must be checked in the selected release; the portal page was
     not sufficient to verify compatibility with these CSVs.
2. **NASA POWER**
   - <https://power.larc.nasa.gov/docs/services/api/>
   - <https://power.larc.nasa.gov/parameters/>
   - Provides documented gridded weather parameters such as temperature,
     relative humidity, and precipitation with time-series queries.
   - It is a weather source, not a crop-label or soil-laboratory dataset.
3. **FAOSTAT crop production, yield, and harvested area**
   - <https://www.fao.org/faostat/en/#data>
   - <https://data.apps.fao.org/catalog/dataset/d24a448b-3b62-4c09-8c1d-4a39bb599876>
   - Provides country/territory and annual crop production statistics with
     documented production, yield, and harvested-area concepts.
   - It is not row-level soil/weather crop-suitability ground truth.
4. **ISRIC SoilGrids**
   - <https://www.isric.org/explore/soilgrids>
   - <https://soilgrids.org/>
   - Provides gridded soil-property estimates by depth and geography.
   - It is not a crop-label dataset and cannot by itself establish observed
     crop suitability or labels.

These sources are candidates for constructing a new documented data product,
not evidence that either local CSV already came from them.

## Compatibility conclusion

The complete field-by-field assessment is in
[dataset_compatibility_matrix.csv](dataset_compatibility_matrix.csv).

For the two local datasets, no feature or target is marked
`VERIFIED COMPATIBLE`. Matching column names are insufficient. The strongest
current conclusion is:

- Schema names: **potentially compatible**
- Numeric representation: **verified as numeric locally**
- Units, measurement methodology, geography, time period, and labels:
  **unknown**
- Combined scientific justification: **not established**

## Strategy recommendation

Choose **D — keep the current model and acquire better-labelled data**.

Evidence:

- The two files have materially different feature-label relationships.
- The independent-only classes cannot be evaluated by the frozen baseline.
- The shared-class evaluation produced 4/416 baseline correct and 0/416
  experimental correct.
- Neither file has the metadata needed to determine whether the differences
  are real population differences, unit mismatches, synthetic generation, or
  target mismatch.
- Existing documentation warns that the original benchmark is not
  real-farm validation.

Do not use the current failed evaluation to tune, relabel, clip, or transform
either source. Obtain a provenance-complete dataset first, then define a
frozen grouped evaluation protocol before any retraining decision.


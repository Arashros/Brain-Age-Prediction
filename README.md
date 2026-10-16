# Brain Age Prediction from MRI Morphometry

An engineering-focused, reproducible machine-learning pipeline for estimating chronological age from structural MRI-derived morphometric features. The project builds on the author's undergraduate work in brain morphometry while using a separately sourced, openly licensed dataset.

## Research question

How accurately can regional structural MRI measurements estimate chronological age in an adult lifespan sample, and how do regularized linear and tree-based regression models compare under participant-level validation?

This is an educational research project. A predicted age is not a clinical measurement or diagnosis.

## Dataset

The planned benchmark is the Dallas Lifespan Brain Study (DLBS), OpenNeuro accession **ds004856**. MRI scan-age fields are in `participants.tsv`; extracted FreeSurfer structural measures are in `derivatives/brainsummary/Template_Structural_MRI.xlsx`. The dataset metadata declares CC0. Cite the dataset paper in any report or presentation:

- Park, D. C. et al. *The Dallas Lifespan Brain Study: A Comprehensive Adult Lifespan Data Set of Brain and Cognitive Aging*. Scientific Data 12, 846 (2025). DOI: 10.1038/s41597-025-04847-7.
- OpenNeuro: ds004856, version 1.3.0, DOI: 10.18112/openneuro.ds004856.v1.3.0.

The raw MRI and source tables are not included in this repository. The structural workbook contains extracted morphometry fields and participant/epoch keys. `prepare_dlbs.py` joins these values to MRI scan age (`AgeMRI_W1`, `AgeMRI_W2`, or `AgeMRI_W3`) from `participants.tsv`. Keep the downloaded source files under `data/raw/` and the prepared table under `data/processed/`; both paths are ignored by Git.

## Methods

- Median-age dummy baseline, Ridge regression, Random Forest, and histogram gradient boosting.
- Participant-grouped holdout split and grouped cross-validation so repeat visits from one participant do not cross the train/test boundary.
- Imputation and scaling are fitted inside each model pipeline, avoiding preprocessing leakage across folds.
- Report MAE, RMSE, and R² for cross-validation and the held-out participant set.
- Save held-out predictions, metrics, and fitted model artifacts under `outputs/` (ignored by Git).

## Initial results

The first run used 957 participant-visits from 464 participants (age 21–97 years) and 274 MRI-derived features. A seeded 80/20 participant-grouped holdout produced 766 training visits from 371 participants and 191 test visits from 93 participants. Five-fold grouped cross-validation was run within the training set. No hyperparameter search was performed; these are initial internal estimates.

| Model | Grouped CV MAE (years) | Test MAE (years) | Test RMSE (years) | Test R² |
|---|---:|---:|---:|---:|
| Median-age baseline | 14.19 | 15.49 | 19.01 | -0.09 |
| Ridge regression | 8.03 | 7.09 | 9.69 | 0.72 |
| Random Forest | 7.71 | 6.98 | 8.70 | 0.77 |
| Histogram gradient boosting | 7.40 | 6.84 | 8.68 | 0.77 |

The participant grouping prevents the same participant's repeated visits from appearing in both training and test sets. This single holdout is not external validation, and the model settings have not been tuned. Treat these results as a reproducible baseline for further analysis, not as evidence of clinical suitability or broad population generalization. The full metrics and held-out predictions are written to `outputs/` when the pipeline runs.

## Repository layout

```text
Brain_Age_Prediction/
├── data/
│   └── processed/       # local input table; ignored by Git
├── outputs/             # metrics, predictions, fitted estimators; ignored by Git
├── src/brain_age/
│   ├── prepare_dlbs.py
│   └── pipeline.py
├── requirements.txt
└── README.md
```

## Run

From this repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/brain_age/prepare_dlbs.py \
  --participants data/raw/ds004856/participants.tsv \
  --structural-file data/raw/ds004856/derivatives/brainsummary/Template_Structural_MRI.xlsx
python src/brain_age/pipeline.py \
  --input data/processed/brain_age_features.csv \
  --target age \
  --group-column participant_id \
  --exclude wave \
  --output-dir outputs
```

The expected source paths are `participants.tsv` and `derivatives/brainsummary/Template_Structural_MRI.xlsx` from the same DLBS snapshot. If the workbook uses different ID or epoch headers, pass `--id-column` and `--epoch-column` to `prepare_dlbs.py`.

For a prepared table whose morphometry fields use hemisphere prefixes, restrict predictors explicitly:

```bash
python src/brain_age/pipeline.py \
  --input data/processed/brain_age_features.csv \
  --feature-prefix lh_ \
  --feature-prefix rh_
```

If there is only one row per participant, the participant ID is still used as the grouping variable. When there are multiple visits, preserve the same ID across visits. Do not include age-derived variables, scan labels, or other target proxies as predictors.

## Interpretation and limitations

Performance depends on the selected cohort, age distribution, FreeSurfer version, feature definitions, and quality-control decisions. The held-out estimate reflects this dataset and split; it is not evidence of clinical readiness or generalization to another scanner or population. Record the exact data version and preprocessing decisions when reporting results.

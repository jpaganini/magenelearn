<div align="center"><img src="figures/logo.png" alt="maGeneLearn" width="600"/></div>

# MaGeneLearn – Bacterial genomics ML pipeline

MaGeneLearn is a modular command-line tool for training, evaluating, and applying
machine-learning models to bacterial genomics feature tables. It chains together a
set of numbered pipeline scripts:

```text
00_split_dataset.py → 01_chisq_selection.py → 02_feature_selection.py →
03_extract_features.py → 04_train_model.py → 05_evaluate_model.py
```

The pipeline supports phylogeny-aware train/test splitting, optional feature
selection with Chi-square, MUVR, or Boruta, multiple classifiers, several
class-balancing strategies, cross-validation, model interpretation, and
prediction on external datasets.

---

## Table of contents

- [Why MaGeneLearn?](#why-magenelearn)
- [1. Installation](#1-installation)
- [2. Test the installation](#2-test-the-installation)
- [3. Command-line overview](#3-command-line-overview)
- [4. Inputs](#4-inputs)
- [5. Training options](#5-training-options)
- [6. Recommended usage of train to save time, memory and headaches](#6-recommended-usage-of-train-to-save-time-memory-and-headaches)
- [7. Test and prediction mode](#7-test-and-prediction-mode)
- [8. Common use cases](#8-common-use-cases)
- [9. Outputs](#9-outputs)
- [10. Citation](#10-citation)
- [11. Contact](#11-contact)

---

## Why MaGeneLearn?

### 1. Phylogeny-aware train/test split

MaGeneLearn can split datasets while accounting for bacterial population
structure. The metadata should contain two levels of grouping:

- **Fine-scale grouping**: outbreak-like or near-identical isolates that should
  not be split across train and test. Examples include EnteroBase HC5, SNP
  clusters, cgMLST clusters, or custom outbreak clusters.
- **Coarse-scale lineage grouping**: broader lineages used to preserve lineage
  composition across train and test. Examples include ST, LINEAGE, EnteroBase
  HC50, or custom lineage labels.

This reduces data leakage and helps produce a more realistic hold-out set.

### 2. Outcome-stratified splitting

In addition to phylogenetic grouping, MaGeneLearn stratifies by the target
outcome/label so class proportions are preserved as much as possible in both
training and test sets.

### 3. Flexible bacterial genomics features

MaGeneLearn can work with binary or dense presence/absence-like feature tables,
including:

- unitigs
- k-mers
- cgMLST or wgMLST one-hot profiles
- accessory genes
- AMR or virulence markers
- combined feature matrices

### 4. Feature reduction

Large bacterial genomics matrices often contain hundreds of thousands or millions
of features. MaGeneLearn can reduce this space using:

- **Chi-square filtering**
- **MUVR**
- **Boruta**

Feature selection can be run once and reused across several models.

### 5. Multiple models and class-balancing strategies

Supported classifiers:

- `RFC` – Random Forest Classifier
- `XGBC` – XGBoost Classifier
- `SVM` – Support Vector Machine
- `LR` – Logistic Regression

Supported sampling strategies:

- `none`
- `random`
- `smote`
- `enn`
- `smoteenn`
- `random_under`

### 6. Model interpretation

MaGeneLearn reports model interpretation outputs after training/evaluation:

- SHAP values for Random Forest and XGBoost models
- permutation importance for SVM models

---

## 1. Installation

Create and activate a conda environment:

```bash
conda create -n magenelearn python=3.9
conda activate magenelearn
```

Install MaGeneLearn from PyPI:

```bash
pip install maGeneLearn
```

After installation, the `maGeneLearn` executable should be available on your
`$PATH`:

```bash
maGeneLearn --help
```

---

## 2. Test the installation

A minimal end-to-end test command:

```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features test/full_train/full_features.tsv \
  --name full_pipe \
  --n-splits 5 \
  --model RFC \
  --chisq \
  --muvr \
  --upsampling random \
  --group-column t5 \
  --label SYMP \
  --lineage-col LINEAGE \
  --k 5000 \
  --n-iter 10 \
  --output-dir full_pipe \
  --n-splits-cv 7
```

For all available options:

```bash
maGeneLearn train --help
maGeneLearn test --help
```

---

## 3. Command-line overview

MaGeneLearn exposes two main commands:

| Command | Purpose |
|---|---|
| `maGeneLearn train` | Train a model end-to-end: split, optional feature selection, model fitting, CV, and evaluation |
| `maGeneLearn test` | Evaluate or apply an already trained model to a test or external dataset |

---

## 4. Inputs

### 4.1 Metadata

The metadata file should be a tab-separated file (`.tsv`) containing at least:

| Column type | Default flag | Description |
|---|---|---|
| Sample ID | `--id-col SRA` | Column used as the isolate/sample identifier |
| Label/outcome | `--label outcome` | Target variable to predict |
| Fine-scale group | `--group-column group` | Grouping column used to avoid leakage across train/test and CV folds |
| Coarse lineage | `--lineage-col LINEAGE` | Lineage/clade column used during train/test splitting |

Example metadata columns:

```text
SRA    outcome    group    LINEAGE
```

In this example you might use:

```bash
--id-col SRA --label outcome --group-column group --lineage-col LINEAGE
```

### 4.2 Feature matrix

The feature matrix should be a tab-separated table where rows correspond to
isolates/samples and columns correspond to genomic features.

Typical examples:

```text
sample_id    feature_1    feature_2    feature_3
isolate_001  0            1            0
isolate_002  1            1            0
```

The sample identifiers should match those in the metadata.

---

## 5. Training options

### 5.1 Required or commonly required options

| Option | Description |
|---|---|
| `--meta-file` | Metadata TSV for an unsplit dataset |
| `--train-meta` | Pre-split training metadata TSV; implies `--no-split` |
| `--test-meta` | Optional pre-split test metadata TSV |
| `--features` | Full feature matrix |
| `--name` | Prefix used for output files |
| `--model` | Final classifier: `RFC`, `XGBC`, `SVM`, or `LR` |
| `--label` | Target/outcome column |
| `--group-column` | Grouping column for grouped splitting/CV |
| `--lineage-col` | Lineage/clade column for Step 00 splitting |
| `--id-col` | Sample ID column |
| `--output-dir` | Output directory |

Use either `--meta-file` for an unsplit dataset or `--train-meta`/`--test-meta`
for pre-split metadata.

### 5.2 Feature-selection options

| Option | Default | Description |
|---|---:|---|
| `--chisq` / `--no-chisq` | off | Run Chi-square feature filtering |
| `--k` | `100000` | Number of top features retained after Chi-square filtering |
| `--chisq-file` | none | Pre-computed Chi-square-filtered matrix; bypasses Step 01 |
| `--muvr` / `--no-muvr` | off | Run MUVR feature selection |
| `--boruta` / `--no-boruta` | off | Run Boruta feature selection |
| `--feature-model` | value of `--model` | Model used inside MUVR/Boruta; choices: `RFC`, `XGBC` |
| `--feature-selection-only` | off | Stop after Step 03 and do not train a final model |

Important rules:

- `--muvr` and `--boruta` require either `--chisq` or `--chisq-file`.
- `--muvr` and `--boruta` also require the original full feature matrix via
  `--features`, because Step 03 extracts the selected features from the full
  matrix.
- Choose either `--muvr` or `--boruta`, not both.
- If running only feature selection, provide `--feature-model` or provide
  `--model` so it can be used as the feature-selection model.

### 5.3 MUVR-specific options

| Option | Default | Description |
|---|---:|---|
| `--dropout-rate` | `0.9` | Proportion of features randomly dropped during MUVR |
| `--muvr-n-repetitions` | `10` | Number of MUVR repetitions |
| `--muvr-n-outer` | `5` | Number of outer MUVR folds |
| `--muvr-n-inner` | `4` | Number of inner MUVR folds |

### 5.4 Boruta-specific options

| Option | Default | Description |
|---|---:|---|
| `--boruta-perc` | `100` | Boruta percentile for shadow feature comparison |
| `--boruta-alpha` | `0.05` | Boruta significance threshold |
| `--boruta-max-iter` | `100` | Maximum number of Boruta iterations |

### 5.5 Model-training options

| Option | Default | Description |
|---|---:|---|
| `--model` | required unless `--feature-selection-only` | Final model: `RFC`, `XGBC`, `SVM`, or `LR` |
| `--upsampling` | `none` | Sampling strategy: `none`, `random`, `smote`, `enn`, `smoteenn`, `random_under` |
| `--n-iter` | `100` | Number of hyperparameter optimization trials |
| `--n-splits-cv` | `7` | Number of grouped CV folds for training/evaluation |
| `--scoring` | `balanced_accuracy` | Metric used to select the best hyperparameters |
| `--n-jobs` | `-1` | Number of parallel jobs |
| `--xgb-policy` | `depthwise` | XGBoost tree growth policy: `depthwise` or `lossguide` |
| `--lr-penalty` | `l2` | Logistic Regression penalty: `l1`, `l2`, or `elasticnet` |

Available scoring metrics:

```text
accuracy
balanced_accuracy
f1
f1_macro
f1_micro
precision
recall
roc_auc
```

### 5.6 Sampling strategies

| Value | Meaning |
|---|---|
| `none` | No explicit resampling |
| `random` | Random over-sampling |
| `smote` | SMOTE over-sampling |
| `enn` | Edited Nearest Neighbours cleaning/under-sampling |
| `smoteenn` | Combined SMOTE over-sampling and ENN cleaning |
| `random_under` | Random under-sampling |

---

## 6. Recommended usage of train to save time, memory and headaches

The recommended workflow is to run heavy feature-selection steps once, then
train several models on the same selected feature matrix. This avoids
recomputing expensive Chi-square/MUVR/Boruta steps for every model and keeps
all downstream model comparisons on exactly the same selected features.

### A) Run splitting and feature selection once

#### Chi-square + MUVR

```bash
maGeneLearn train \
  --feature-selection-only \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features test/full_train/full_features.tsv \
  --name STEC \
  --label SYMP \
  --group-column t5 \
  --lineage-col LINEAGE \
  --chisq \
  --muvr \
  --feature-model RFC \
  --k 5000 \
  --n-splits 5 \
  --n-jobs -1 \
  --output-dir selected_features_muvr
```

#### Chi-square + Boruta

```bash
maGeneLearn train \
  --feature-selection-only \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features test/full_train/full_features.tsv \
  --name STEC \
  --label SYMP \
  --group-column t5 \
  --lineage-col LINEAGE \
  --chisq \
  --boruta \
  --feature-model RFC \
  --boruta-max-iter 500 \
  --boruta-perc 95 \
  --boruta-alpha 0.1 \
  --k 5000 \
  --n-splits 5 \
  --n-jobs -1 \
  --output-dir selected_features_boruta
```

These commands create final feature tables in:

```text
<output-dir>/03_final_features/<name>_train.tsv
<output-dir>/03_final_features/<name>_test.tsv
```

### B) Train several models using the selected features

#### Random Forest with random over-sampling

```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features_muvr/03_final_features/STEC_train.tsv \
  --features-test selected_features_muvr/03_final_features/STEC_test.tsv \
  --name STEC_muvr \
  --model RFC \
  --upsampling random \
  --n-iter 100 \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/RFC_random
```

#### XGBoost with SMOTEENN

```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features_muvr/03_final_features/STEC_train.tsv \
  --features-test selected_features_muvr/03_final_features/STEC_test.tsv \
  --name STEC_muvr \
  --model XGBC \
  --upsampling smoteenn \
  --xgb-policy lossguide \
  --n-iter 100 \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/XGBC_smoteenn
```

#### SVM with ENN

```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features_muvr/03_final_features/STEC_train.tsv \
  --features-test selected_features_muvr/03_final_features/STEC_test.tsv \
  --name STEC_muvr \
  --model SVM \
  --upsampling enn \
  --n-iter 100 \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/SVM_enn
```

#### Logistic Regression with random under-sampling

```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features_muvr/03_final_features/STEC_train.tsv \
  --features-test selected_features_muvr/03_final_features/STEC_test.tsv \
  --name STEC_muvr \
  --model LR \
  --lr-penalty l2 \
  --upsampling random_under \
  --n-iter 100 \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/LR_random_under
```

---

## 7. Test and prediction mode

Use `maGeneLearn test` to evaluate or apply an already trained model.

### 7.1 Required options

| Option | Description |
|---|---|
| `--model-file` | `.joblib` model produced by `maGeneLearn train` |
| `--name` | Prefix for output files |
| `--output-dir` | Output directory |

You must provide exactly one of:

| Option | Use case |
|---|---|
| `--features-test` | A ready-to-use filtered feature table |
| `--features` | A full external feature matrix that must be filtered using `--feature-file` |

### 7.2 Optional test-mode options

| Option | Description |
|---|---|
| `--feature-file` | Selected-feature file used to filter a full external feature matrix |
| `--test-metadata` | Metadata for labelled external data |
| `--predict-only` | Produce predictions without computing performance metrics |
| `--label` | Label column for labelled test data |
| `--group-column` | Group column for grouped metrics |
| `--scoring` | Scoring metric |
| `--skip-svm-importance` | Skip permutation importance for SVM models |

### 7.3 Evaluate an existing hold-out set

Use this when you already have a filtered test matrix, for example from
`03_final_features/`.

```bash
maGeneLearn test \
  --model-file runs/RFC_random/04_model/STEC_muvr_RFC_random.joblib \
  --features-test selected_features_muvr/03_final_features/STEC_test.tsv \
  --name STEC_muvr_RFC_random \
  --output-dir runs/RFC_random \
  --label SYMP \
  --group-column t5
```

### 7.4 Evaluate labelled external isolates

Use this when you have a full external feature matrix and labels are available.

```bash
maGeneLearn test \
  --model-file runs/RFC_random/04_model/STEC_muvr_RFC_random.joblib \
  --feature-file selected_features_muvr/02_muvr/STEC_muvr_RFC_min.tsv \
  --features test/external_data/full_features_external.tsv \
  --test-metadata test/external_data/metadata_external.tsv \
  --name External_STEC_RFC_random \
  --output-dir runs/RFC_random_external \
  --label SYMP \
  --group-column t5
```

### 7.5 Predict unlabelled external isolates

Use this when no ground-truth labels are available.

```bash
maGeneLearn test \
  --predict-only \
  --model-file runs/RFC_random/04_model/STEC_muvr_RFC_random.joblib \
  --feature-file selected_features_muvr/02_muvr/STEC_muvr_RFC_min.tsv \
  --features test/external_data/full_features_unlabelled.tsv \
  --name External_STEC_RFC_random \
  --output-dir runs/RFC_random_predict
```

Prediction-only mode creates:

```text
<output-dir>/07_test_eval/<name>_test_predictions.tsv
```

The prediction table contains the predicted class and, when available, class
probability columns.

### 7.6 Test-mode naming

The test command infers model type, sampling strategy, and LR penalty from the
model filename. For example:

```text
STEC_muvr_RFC_random.joblib
STEC_boruta_XGBC_smoteenn.joblib
STEC_boruta_LR_random_under_l2.joblib
```

If `--name` already contains the model/sampling suffix, MaGeneLearn does not add
it again. This prevents duplicated names such as:

```text
STEC_RFC_random_under_random_under_test_predictions.tsv
```

and fallback names such as:

```text
STEC_NA_none_test_predictions.tsv
```

---

## 8. Common use cases

### 8.1 Feature selection without splitting

Use this when you already have pre-split metadata.

#### Pre-split Chi-square + MUVR

```bash
maGeneLearn train \
  --no-split \
  --feature-selection-only \
  --train-meta test/skip_split/train_metadata.tsv \
  --test-meta test/skip_split/test_metadata.tsv \
  --features test/skip_split/full_features.tsv \
  --name STEC \
  --label SYMP \
  --group-column t5 \
  --chisq \
  --muvr \
  --feature-model RFC \
  --k 5000 \
  --n-jobs -1 \
  --output-dir selected_features_split_muvr
```

#### Pre-split Chi-square + Boruta

```bash
maGeneLearn train \
  --no-split \
  --feature-selection-only \
  --train-meta test/skip_split/train_metadata.tsv \
  --test-meta test/skip_split/test_metadata.tsv \
  --features test/skip_split/full_features.tsv \
  --name STEC \
  --label SYMP \
  --group-column t5 \
  --chisq \
  --boruta \
  --feature-model RFC \
  --boruta-max-iter 500 \
  --boruta-perc 95 \
  --boruta-alpha 0.1 \
  --k 5000 \
  --n-jobs -1 \
  --output-dir selected_features_split_boruta
```

### 8.2 Skip Chi-square and run MUVR or Boruta on an already-filtered matrix

If your starting feature matrix is already reasonably small, you can skip Step
01 by passing the same matrix to both `--features` and `--chisq-file`.

#### Already-filtered matrix + MUVR

```bash
maGeneLearn train \
  --feature-selection-only \
  --meta-file test/skip_chi/metadata.tsv \
  --chisq-file test/skip_chi/filtered_features.tsv \
  --features test/skip_chi/filtered_features.tsv \
  --name STEC \
  --muvr \
  --feature-model RFC \
  --group-column t5 \
  --label SYMP \
  --lineage-col LINEAGE \
  --output-dir skip_chi_muvr
```

#### Already-filtered matrix + Boruta

```bash
maGeneLearn train \
  --feature-selection-only \
  --meta-file test/skip_chi/metadata.tsv \
  --chisq-file test/skip_chi/filtered_features.tsv \
  --features test/skip_chi/filtered_features.tsv \
  --name STEC \
  --boruta \
  --feature-model RFC \
  --boruta-max-iter 500 \
  --boruta-perc 95 \
  --boruta-alpha 0.1 \
  --group-column t5 \
  --label SYMP \
  --lineage-col LINEAGE \
  --output-dir skip_chi_boruta
```

---

## 9. Outputs

### 9.1 Feature-selection outputs

A feature-selection-only run creates:

```text
<output-dir>/
├── 00_data_split/
│   ├── <name>_train.tsv
│   └── <name>_test.tsv
├── 01_chisq/
│   └── <name>_top<k>_features.tsv
├── 02_muvr/ or 02_boruta/
│   └── <name>_<method>_<feature-model>_min.tsv
└── 03_final_features/
    ├── <name>_train.tsv
    └── <name>_test.tsv
```

### 9.2 Training outputs

A full training run creates:

```text
<output-dir>/
├── 04_model/
│   └── <name>_<model>_<sampling>.joblib
├── 05_cv/
│   └── cross-validation and hyperparameter optimization outputs
├── 06_train_eval/
│   ├── train prediction/probability tables
│   ├── classification reports
│   ├── MCC/AUPRC metrics
│   ├── confusion matrices
│   └── SHAP or permutation-importance outputs
└── 07_test_eval/
    └── hold-out test evaluation outputs, when a test matrix is available
```

For Logistic Regression, model files include the LR penalty:

```text
<name>_LR_<sampling>_<lr-penalty>.joblib
```

For example:

```text
STEC_boruta_LR_random_under_l2.joblib
```

### 9.3 Test outputs

Labelled test mode creates evaluation outputs in:

```text
<output-dir>/07_test_eval/
```

Prediction-only mode creates:

```text
<output-dir>/07_test_eval/<name>_test_predictions.tsv
```

---

## 10. Citation

If you use MaGeneLearn, please cite:

> Predicting clinical outcome of *Escherichia coli* O157:H7 infections using
> explainable Machine Learning  
> https://doi.org/10.1101/2025.06.05.25329036

Please also cite the tools/methods relevant to your analysis:

- Optuna:  
  https://doi.org/10.48550/arXiv.1907.10902

- scikit-learn:  
  https://scikit-learn.org/stable/about.html#citing-scikit-learn

- MUVR, if used for feature selection:  
  https://doi.org/10.1093/bioinformatics/bty710

- SHAP, if used for feature interpretation:  
  https://doi.org/10.48550/arXiv.1705.07874

---

## 11. Contact

Questions or issues? Please contact:

```text
j.a.paganini@uu.nl
```

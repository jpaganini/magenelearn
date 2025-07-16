 <div align="center"><img src="figures/logo.png" alt="maGeneLearn" width="600"/></div>

# MaGeneLearn  – Bacterial genomics ML pipeline
MaGeneLearn is a modular CLI that chains together a set of numbered Python
scripts (`00_split_dataset.py → 05_evaluate_model.py`) to train and evaluate
machine-learning models from (potentially huge) presence/absence tables.

The wrapper exposes **two** high-level commands:

| Command | What it does |
|---------|--------------|
| `magene-learn train` | end-to-end model building (split → *optional* feature-selection → fit → CV → eval) |
| `magene-learn test`  | evaluate an already–trained model on an external set ( **no CV** ) |

---

## 1 Installation

```bash
conda create -n magenelearn python=3.9
conda activate magenelearn
pip install magenelearn
```
now `maGeneLearn` should be on your $PATH

## 2 Test the installation
```bash
maGeneLearn --help
maGeneLearn train --meta-file test/full_train/2023_jp_meta_file.tsv --features test/full_train/full_features.tsv --name full_pipe --model RFC --chisq --muvr --upsampling random --group-column t5 --label SYMP --lineage-col LINEAGE --k 5000 --n-iter 10 --output-dir full_pipe
```


## 3 Command-line reference

```bash
maGeneLearn train [OPTIONS]               # model building pipeline
maGeneLearn test  [OPTIONS]               # evaluate existing model
maGeneLearn --help                        # top-level help
maGeneLearn <subcmd> --help               # help for a sub-command
```


## 4 · train – build a model end-to-end

### Always Required

| flag          | file            | purpose                                              |
| ------------- | --------------- | ---------------------------------------------------- |
| `--meta-file` | TSV             | sample metadata with **label** & **group** columns   |
| `--features`  | TSV             | *full* k-mer matrix (rows = isolates, cols = k-mers) |
| `--name`      | str             | prefix for every artefact                            |
| `--model`     | `RFC` \| `XGBC` | classifier for step 04                               |

### Frequently useful

| flag                 | default                 | effect                                                 |
| -------------------- | ----------------------- | -------------------------------------------------------|
| `--features2`        | –                       | merge a second k-mer matrix                            |
| `--no-split`         | off                     | skip **00** (expects `<name>_train/_test.tsv` ready)   |
| `--chisq` 	       | off                     | run Step 01 Chi² filtering                             |
| `--muvr`             | off                     | run Step 02 MUVR                                       |
| `--muvr-model`       | =`--model`              | algorithm used **inside** MUVR                         |
| `--features-train`   | –                       | pre-built training matrix – skips 00-03                |
| `--features-test`    | –                       | pre-built hold-out matrix – skips 07                   |
| `--upsampling`       | `none / smote / random` |                                                        |
| `--n-splits`         | 5                       | CV folds for training                                  |
| `--scoring`          | balanced_accuracy       | Metric used to select the best hyperparameters         |
| `--output-dir`       | timestamp               | root of the run                                        |
| `--lineage-col`      | LINEAGE                 | Column name. Use to split the data with stratification |
| `--output-dir`       | timestamp               | root of the run                                        |
| `--dry-run`          | –                       | print commands, do nothing                             |

### Typical flavours

* **Full pipeline (split → Chi² → MUVR → (Upsampling) + model optimization)**
```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features  test/full_train/full_features.tsv \
  --name STEC \
  --muvr-model XGBC \
  --model RFC \
  --chisq --muvr \
  --upsampling smote\
  --group-column t5
  --label SYMP
  --lineage-col LINEAGE 
  --k 5000 
  --n-iter 10
```

* **Skip Chi² (use an already-filtered matrix, still run MUVR)**  
  You already produced a Chi²-filtered table elsewhere (or manually picked  
  a subset of features) and just want MUVR + model training.

```bash
  maGeneLearn train \
  --meta-file test/skip_chi/2023_jp_meta_file.tsv \
  --chisq-file test/skip_chi/chisq_reduced.tsv \
  --features test/skip_chi/full_features.tsv \
  --name full_pipe \
  --model XGBC \
  --muvr
  --muvr-model RFC \
  --upsampling smote \
  --group-column t5 \
  --label SYMP \
  --lineage-col LINEAGE \
  --output-dir skip_chi_test
```

If the full matrix is small enough and no chisq step is needed, the full matrix can be passed to both --features and --chisq-file arguments.

* **Already split metadata (--no-split)
```bash
maGeneLearn train 
  --no-split \
  --train-meta test/skip_split/train_metadata.tsv \
  --test-meta test/skip_split/test_metadata.tsv \
  --features test/skip_split/full_features.tsv \
  --name STEC \
  --model RFC \
  --chisq --muvr \
  --label SYMP \ 
  --group-column t5 \ 
  --k 2000 \
  --n-iter 10
```

## 5 · test – evaluate saved model

* **Three ways to give test features:**

| scenario                                                      | flags you pass                                       |
| ------------------------------------------------------------- | ---------------------------------------------------- |
| **A.    Evaluate performance on a test-set**  | `--features-test` `--label` `--group-column`                                     |
| **B.    Classifying new samples WITHOUT labels**  | `--features` (full) `--muvr-file` `--predict-only`                                  |
| **C.    Classifying new samples WITH labels**            | `--features` (full)  `--muvr-file` `--test-metadata` `--label` `--group-column` |


* **Required**
  
| flag           | meaning                        |
| -------------- | ------------------------------ |
| `--model-file` | `.joblib` from the *train* run |
| `--name`       | prefix for outputs             |

### **Scenario A - Evaluate performance on a test-set***

In this scenario you have already run a **full** training pipeline using `maGeneLearn train`. Now, you want to evaluate the performance on the test-set.
After running `maGeneLearn train`, your model file will be located in `<output-dir>/04_model/<name>.joblib`. And your features-test matrix will be located in `<output-dir>/03_final_features/<name>_test.tsv`. We will use these files to evaluate performance.

Following on the installation example from Section 2 of this user-guide, we can evaluate the performance on the test-set using the following command: 

```bash
maGeneLearn test \
  --model-file full_pipe/04_model/full_pipe_RFC_random.joblib \
  --features-test full_pipe/03_final_features/full_pipe_test.tsv \
  --name full_pipe\
  --output-dir full_pipe\
  --label SYMP \
  --group-column t5
```

This will create a new directory `/07_test_eval` inside the existing directory `/full_pipe`. In this directory you'll find the predictions on each isolate from the test set, the evaluation metrics and the SHAP importance values.

### **Scenario B - Classifying new samples WITHOUT labels**

In this scenario, you have trained your ML-model using any variation of the `maGeneLearn train` pipeline. Now, you have a new set of isolates for which you would like to make predictions. This is probably the most common use case in a practical setting. 

Again, in the example run below we use the model created in **Section 2** and one of the test files included in the git repo. 

```bash
maGeneLearn test \
  --predict-only \
  --model-file full_pipe/04_model/full_pipe_RFC_random.joblib \
  --features test/full_train/full_features.tsv \
  --muvr-file full_pipe/02_muvr/full_pipe_muvr_RFC_min.tsv \
  --name new_test \
  --output-dir predict_only_test \
```
This command will create two new directories:

1- `<output-dir>/03_final_features`: This directory contains a presence/absence file with the features used to train the model.

2- `<output-dir>/07_test_eval`: This directory you'll find a file with the predictions of each new isolate.

### **Scenario C - Classifying new samples WITH labels**

In this scenario, you have trained your ML-model using any variation of the `maGeneLearn train` pipeline. Now, you have a new set of isolates for which you would like to make predictions and evaluate the performance. This probably occurs if you want to perform an external validation of your model, with a distinct dataset.

In the example run below we use the model created in **Section 2** and one of the test files included in the git repo. 

```bash
maGeneLearn test \
  --model-file full_pipe/04_model/full_pipe_RFC_random.joblib \
  --features test/full_train/full_features.tsv \
  --muvr-file full_pipe/02_muvr/full_pipe_muvr_RFC_min.tsv \
  --test-metadata test/full_train/2023_jp_meta_file.tsv \
  --name independent_test \
  --output-dir independent_test \
  --label SYMP \
  --group-column t5
```
This command will create two new directories:

1- `<output-dir>/03_final_features`: This directory contains a presence/absence file with the features used to train the model.

2- `<output-dir>/07_test_eval`: This directory you'll find a file with the predictions of each new isolate, SHAP values and evaluation metrics.


## 6 · Contact

Do you have any doubts? Please contact me at: j.a.paganini@uu.nl.



















# MaGeneLearn  – k-mer based bacterial genomics ML pipeline
MaGeneLearn is a modular CLI that chains together a set of numbered Python
scripts (`00_split_dataset.py → 05_evaluate_model.py`) to train and evaluate
machine-learning models from (potentially huge) k-mer count tables.

The wrapper exposes **two** high-level commands:

| Command | What it does |
|---------|--------------|
| `magene-learn train` | end-to-end model building (split → *optional* feature-selection → fit → CV → eval) |
| `magene-learn test`  | evaluate an already–trained model on an external set ( **no CV** ) |

---

## 1 Installation

```bash
git clone <repo>
cd maGeneLearn
pip install -e .
```
# now `magene-learn` should be on your $PATH

### 1.1 Test the installation
```bash
maGeneLearn --help
maGeneLearn train --meta-file test/full_train/2023_jp_meta_file.tsv --features test/full_train/full_features.tsv --name full_pipe --model RFC --chisq --muvr --upsampling random --group-column t5 --label SYMP --lineage-col LINEAGE --k 5000 --n-iter 10
```


## 2 Command-line reference

```bash
maGeneLearn train [OPTIONS]               # model building pipeline
maGeneLearn test  [OPTIONS]               # evaluate existing model
maGeneLearn --help                        # top-level help
maGeneLearn <subcmd> --help               # help for a sub-command
```


## 3 · train – build a model end-to-end

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

* **Full pipeline (split → Chi² → MUVR → SMOTE + RFC)
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
  a k-mer subset) and just want MUVR + model training.

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

## 4 · test – evaluate saved model

* **Two mutually exclusive ways to give test features:

| scenario                                          | flags you pass                                       |
| ------------------------------------------------- | ---------------------------------------------------- |
| **A. raw full matrix** (filter k-mers on the fly) | `--features` (full)  `--muvr-file` `--test-metadata` |
| **B. ready matrix** (pre-filtered)                | `--features-test`                                    |


* **Required
| flag           | meaning                        |
| -------------- | ------------------------------ |
| `--model-file` | `.joblib` from the *train* run |
| `--name`       | prefix for outputs             |


```bash
magene-learn test \
  --model-file results/04_model/STEC_RFC_none.joblib \
  --features   data/kmers.tsv \
  --muvr-file  results/02_muvr/STEC_muvr_RFC_min.tsv \
  --test-metadata data/meta_external.tsv \
  --name extA
```

```bash
magene-learn test \
  --model-file results/04_model/STEC_RFC_none.joblib \
  --features-test results/03_final/STEC_test.tsv \
  --name extB
```


















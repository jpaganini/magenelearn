 <div align="center"><img src="figures/logo.png" alt="maGeneLearn" width="600"/></div>

# MaGeneLearn  – Bacterial genomics ML pipeline
MaGeneLearn is a modular CLI that chains together a set of numbered Python
scripts (`00_split_dataset.py → 05_evaluate_model.py`) to train and evaluate
machine-learning models from presence/absence tables.

## Table of contents
- [Why MaGeneLearn?](#why-magenelearn)
- [1 Installation](#1-installation)
- [2 Test the installation](#2-test-the-installation)
- [3 Command-line reference](#3-command-line-reference)
- [4 · Inputs for training a new model](#4--inputs-for-training-a-new-model)
- [5 · Most common flags for training](#5--most-common-flags-for-training)
- [6 · Recommended usage of train to save time, memory and headaches](#6--recommended-usage-of-train-to-save-time-memory-and-headaches)
- [7 · Evaluate or predict using test mode](#7--evaluate-or-predict-using-test-mode)
- [8 · Different flavours](#8--different-flavours)
  - [8.1) Select features without splitting](#81-select-features-without-splitting)
  - [8.2) Skip Chi² (use an already-filtered matrix, still run MUVR)](#82-skip-chi-use-an-already-filtered-matrix-still-run-muvr)
- [9 · Cite](#9--cite)
- [10 · Contact](#10--contact)


## Why MaGeneLearn?

* **1) Phylogeny-aware train/test split**

Provide two levels of phylogenetic clustering in your metadata:

*	*Outbreak-like clustering (fine scale)*: groups of near-identical isolates that must not be split across train/test (prevents data leakage). Examples: EnteroBase HC5, or your own SNP/cgMLST cluster IDs.

*	*Higher-level clustering (coarser scale)*: used for stratification so both train and test retain similar composition across lineages. Examples: ST, LINEAGE, EnteroBase HC50.
MaGeneLearn keeps each outbreak-like cluster entirely in one split and stratifies by the higher-level clusters.

* **2) Outcome-stratified split** 

In addition to phylogeny, the split is stratified by the outcome/label, preserving class proportions in both train and test.

* **3)Efficient hyperparameter optimization (Optuna)**
Uses Optuna’s state-of-the-art samplers and pruning to find strong settings quickly for each supported model.

* **4)Feature importance for interpretability**

After training, MaGeneLearn reports feature importance via SHAP (XGBoost, Random Forest) or permutation importance (SVM), helping you “open the black box” and link predictive signals to biology.

* **5) Flexible inputs**
Any binary presence/absence features are supported: unitigs, k-mers, one-hot cgMLST profiles, accessory genes—or combinations thereof.

* **6)Built-in feature reduction**
Large initial feature spaces often contain noise. MaGeneLearn can reduce to the most informative set using Chi-square and/or MUVR/Boruta, improving signal-to-noise before model fitting.

**Metadata requirements (for splitting):**

*	A column with outcome (**label**).

*	A fine-scale cluster column (**group**) (outbreak-like; e.g., HC5/SNP/cgMLST cluster).

*	A coarse-scale cluster column for stratification (**lineage**) (e.g., ST/LINEAGE/HC50).



---

## 1 Installation

```bash
conda create -n magenelearn python=3.9
conda activate magenelearn
pip install maGeneLearn
```
now `maGeneLearn` should be on your $PATH

## 2 Test the installation
```bash
maGeneLearn --help
maGeneLearn train --meta-file test/full_train/2023_jp_meta_file.tsv --features test/full_train/full_features.tsv --name full_pipe --n-splits 5 --model RFC --chisq --muvr --upsampling random --group-column t5 --label SYMP --lineage-col LINEAGE --k 5000 --n-iter 10 --output-dir full_pipe --n-splits-cv 7
```


## 3 Command-line reference

The wrapper exposes **two** high-level commands:

| Command | What it does |
|---------|--------------|
| `magene-learn train` | end-to-end model building (split → *optional* feature-selection → fit → CV → eval) |
| `magene-learn test`  | evaluate an already–trained model on an external set ( **no CV** ) |


## 4 · Inputs for training a new model

* The **train** command will always require at least two inputs:

| File          | format            | description                                              |
| ------------- | --------------- | ---------------------------------------------------- |
| `metadata file` | TSV             | sample metadata with **label**, **group** and **lineage** columns   |
| `features file`  | TSV             | *full* presence/absence matrix (rows = isolates, cols = features) |


## 5 · Most common flags for training

### Always Required

| flag          | file            | purpose                                              |
| ------------- | --------------- | ---------------------------------------------------- |
| `--meta-file` | TSV             | pass the metadata file                               |
| `--features`  | TSV             | pass the presence/absence matrix                     |
| `--name`      | str             | prefix for every output genereated                   |
| `--model`     | `RFC` \| `XGBC` \| `SVM` \| `LR` | classifier for step 04              |
| `--label`     |  outcome        | Column name containing the target varible     |   
| `--group-column` |  group       | Column name for grouped split data (e.g., outbreak-like cluster) |
| `--n-splits-cv`  | 7            | Number of CV folds to optimize hyperparamter and to evaluate model performance on the training set  |
| `--n-iter`    | 100     | Number of optuna trials for hyperparameters optimization     |


### Frequently useful

| flag                 | default                 | effect                                                 |
| -------------------- | ----------------------- | -------------------------------------------------------|
| `--no-split`         | off                     | skip **00** (expects `<name>_train/_test.tsv` ready)   |
| `--chisq` 	       | off                     | run Chi² filtering                             |
| `--k` 	       | 100000                  | How many features will be selected by Chi² filtering   |
| `--muvr`             | off                     | run Step 02 feature reduction. Requires --chisq or --chisq-file and the original full matrix via --features |
| `--boruta`           | off			 | run Step 02 feature reduction. Requires --chisq or --chisq-file and the original full matrix via --features |
| `--chisq-file`       | –                       | Pre-computed Chi²-reduced matrix (bypasses Step 01).   |
| `--feature-model`    | =`--model`              | algorithm used **inside** MUVR/Boruta                  |
| `--dropout-rate`     | 0.9                     | Randomly drop a fraction of features during MUVR (stability/regularization) |
| `--boruta-perc`      | 100                     | BorutaPercentile for shadow feature comparison         |
| `--boruta-alpha`     | 0.05                    | The significance threshold for Boruta                  |
| `--boruta-max-iter`  | 100                     | Maximum number of iterations performed for Boruta      |
| `--feature-selection-only` | off               | Run through Step 03 and **exit** (no model fitting).   | 
| `--features-train`   | –                       | pre-built training matrix – skips 00-03. This matrix should contain the final features that will be used to train the model and also the group and outcome columns.                |
| `--features-test`    | –                       | pre-built hold-out matrix. To be used within the "test" mode,                    |
| `--upsampling`       | `none` | Upsampling strategies `none / smote / random / enn / smoteenn / random_under`                          |
| `--n-splits`         | 5                       | Number of folds to create training/test splits. A value of 5 will be equal to do a 80/20 split |
| `--scoring`          | balanced_accuracy       | Metric used to select the best hyperparameters (accuracy, balanced_accuracy, f1, f1_macro, f1_micro, precision, recall, roc_auc).         |
| `--lineage-col`      | LINEAGE                 | Column name. Use to split the data with stratification |
| `--id-co`            | SRA                     | Column name to use as the sample ID/index.             |
| `--output-dir`       | timestamp               | root of the run                                        |
| `--xgb-policy`      | depthwise                 | XGBoost tree growth policy: `depthwise | lossguide` |
| `--lr-penalty`      | l2                 | LR penalty: 'l1 | l2 | elasticnet' |
| `--dry-run`          | –                       | print commands, do nothing                             |

### For a full list of tunnable flags, please run: 

```bash
maGeneLearn train --help     
maGeneLearn test --help        
```

## 6 · Recommended usage of train to save time, memory and headaches

**Goal: avoid re-running heavy selection steps for each model.**

* **A) Split your data into training and test and run feature selection once**
Use --chisq and/or --muvr with --feature-selection-only to stop after selecting the most predictive features. 

#### Runs split → Chi² → MUVR → FINAL FEATURES, then exits
```bash
maGeneLearn train \
  --feature-selection-only \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features  test/full_train/full_features.tsv \
  --name STEC \
  --label SYMP \
  --group-column t5 \
  --lineage-col LINEAGE \
  --chisq --muvr \
  --muvr-model RFC \
  --k 5000 \
  --n-splits 5 \
  --n-jobs -1 \
  --output-dir selected_features \
  
```

This command will create the following output:

```
<output-dir>/
  00_split/
  	${name}_test.tsv  #Metadata file with the isolates selected for testing
  	${name}_train.tsv  #Metadata file with the isolates selected for training 
  	 
  01_chisq/
  	${name}_pvalues_features.tsv  #DFwith dimensions (train_isolates x features) with p-value<=0.05
  	${name}_pvalues.tsv  #Two-column DF, containing p-values for all features
  	${name}_top${k}_features.tsv #DF with dimensions (train_isolates x features) with top-K features based on Chi² scire 
      
  02_feature_selection/
  	${name}_muvr_${muvr-model}_max.tsv #DF with dimensions (train_isolates x features), with features being the maximumum number of informative features after MUVR
  	${name}_muvr_${muvr-model}_mid.tsv #DF with dimensions (train_isolates x features), with features being the medium number of informative features after MUVR
  	${name}_{feature-reduction}_${feature-model}_min.tsv #DF with dimensions (train_isolates x features), with features being the minimum number of informative features after MUVR
      
  03_final_features/
  	${name}_train.tsv #DF similar to those of 02_feature_selection, but fused with label and group columns.
  	${name}_test.tsv #Same as above but for test isolates     

```

* **B) Train multiple models on the same selected features**
For each algorithm you want to compare (RFC, XGBC, SVM, LR), launch train runs that reuse the already-materialized feature tables. This avoids recomputing Chi²/MUVR and ensures a fair comparison across models (same feature set). Required labels will include:

--features-train, the same --group-column, --label, different --model (and --lr-penalty / --xgb-policy as needed).



**RFC (random upsampling)**
```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features/03_final_features/STEC_train.tsv \
  --name STEC \
  --model RFC \
  --upsampling random \
  --n-iter 10 \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/RFC
```

**XGBoost (lossguide)**
```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features/03_final_features/STEC_train.tsv \
  --name STEC \
  --model XGBC \
  --n-iter 10 \
  --upsampling none \
  --xgb-policy lossguide \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/XGB
```

**SVM (SMOTE upsampling)**
```bash
maGeneLearn train \
  --meta-file test/full_train/2023_jp_meta_file.tsv \
  --features-train selected_features/03_final_features/STEC_train.tsv \ 
  --name STEC\
  --model SVM \
  --upsampling smote 
  --n-iter 10 \
  --label SYMP \
  --group-column t5 \
  --n-splits-cv 7 \
  --scoring balanced_accuracy \
  --n-jobs -1 \
  --output-dir runs/SVM
```

Each of these commands will create the following outputs:

```
<output-dir>/
  04_model/
  	${name}_${model}_${upsampling}.joblib   #File containg the model after hyperparameter optimization via CV. 	 
  	 
  05_cv/
  	CV_${name}_${model}_${upsampling}.tsv  #DF with all Optuna optimization and CV results
  	 
  06_train_eval/
	${name}_${model}_${upsampling}_train_prediction_probabilities.tsv  #DF containing the ground truth of every training isolate and their corresponding prediction obtained with the model, when the isolate was part of the CV-test fold. Also it contains the probability of each class assigned by the model.
  	${name}_${model}_${upsampling}_train_classification_report.tsv  #Classification report based on aggregating CV test-folds based on the training data. Includes accuracy, f1-score, recall, precision and macro- and micro-avg of these metrics.
  	${name}_${model}_${upsampling}_train_mcc_auprc.tsv	#Extra evaluatio metrics: MCC and AUPRC.
  	${name}_${model}_${upsampling}_train_confusion_matrix.tsv  #Confusion matrix of the classificaiton of train isolates
	${name}_${model}_${upsampling}_train_confusion_matrix.png  #Same as above, but image version.
	${name}_${model}_${upsampling}_train_<class>_shap_values.tsv  #For each class, a file containing the SHAP values for each feature per isolate.
  	${name}_${model}_${upsampling}_train_<class>_shap_summary.png #For each class, a plot showing the 20 most important features according to shap value.
```


## 7 · Evaluate or predict using test mode**

* **Required**
  
| flag           | meaning                        |
| -------------- | ------------------------------ |
| `--model-file` | `.joblib` from the *train* run |
| `--name`       | prefix for outputs             |


### **Three ways to use the test mode:**


*  **1) Predicting isolates from your original dataset**

In this scenario you have already run a **full** training pipeline using `maGeneLearn train`. Now, you want to evaluate the performance on the test-set.

After running `maGeneLearn train`, your model file will be located in `<output-dir>/04_model/<name>.joblib`. And your features-test matrix will be located in `<output-dir>/03_final_features/<name>_test.tsv`. We will use these files to evaluate performance.

```bash
maGeneLearn test \
  --model-file runs/RFC/04_model/STEC_RFC_random.joblib \
  --features-test selected_features/03_final_features/STEC_test.tsv \
  --name STEC \
  --output-dir runs/RFC \
  --label SYMP \
  --group-column t5
```

The output from this test command will be:

```
<output-dir>/
  07_test_eval/
	${name}_${model}_${upsampling}_test_prediction_probabilities.tsv  #DF containing the ground truth of every isolate in the test set and their corresponding prediction obtained with the model. Also it contains the probability of each class assigned by the model.
  	${name}_${model}_${upsampling}_test_classification_report.tsv  #Classification report based on the test isolates. Includes accuracy, f1-score, recall, precision and macro- and micro-avg of these metrics.
  	${name}_${model}_${upsampling}_test_mcc_auprc.tsv	#Extra evaluatio metrics: MCC and AUPRC.
  	${name}_${model}_${upsampling}_test_confusion_matrix.tsv  #Confusion matrix of the classificaiton of test isolates
	${name}_${model}_${upsampling}_test_confusion_matrix.png  #Same as above, but image version.
```

*  **2)Predict on NEW labelled-isolates**  

In this scenario, you have trained your ML-model using any variation of the `maGeneLearn train` pipeline. Now, you have a new set of isolates for which you would like to make predictions and evaluate the performance. This probably occurs if you want to perform an external validation of your model or evaluate generalizability.

For this you will use the following flags and files:

`--features`: Pass a matrix (in tsv format) that contains the raw features of new samples.

`--feature-file`: You will pass the features that were used for training the model. Most of the times, these are features obtained after a feature selection step. This file will be used to filter the relevant data for the model.


```bash
maGeneLearn test \
  --model-file runs/RFC/04_model/STEC_RFC_random.joblib \
  --feature-file selected_features/02_muvr/STEC_muvr_RFC_min.tsv \
  --features test/external_data/full_features_external.tsv \
  --test-metadata test/external_data/metadata_external.tsv \
  --name External_STEC \
  --output-dir runs/RFC \
  --label SYMP \
```

This will create the same outputs as above.

*  **3)Predict on new unlabelled-isolates** 

In this scenario, you have trained your ML-model using any variation of the `maGeneLearn train` pipeline. Now, you have a new set of isolates for which you would like to make predictions. This is probably the most common use case in a practical setting.

You will require the `--predict-only` flag:

```bash
maGeneLearn test \
  --predict-only \
  --model-file runs/RFC/04_model/STEC_RFC_random.joblib \
  --feature-file selected_features/02_muvr/STEC_muvr_RFC_min.tsv \
  --features test/external_data/full_features_unlabelled.tsv \
  --name unlabelled_isolates \
  --output-dir runs/RFC
```

This will create the same outputs as above, except that it will not evaluate the performance of the model, because there is not a ground truth for these isolates.



## 8 · Different flavours


* **8.1) Select features without splitting** 

In this scenario, you have already split your data. Now you would like to select the most important features to later train multiple models.
Use --chisq and/or --muvr with --feature-selection-only to stop after selecting the most predictive features. 


#### Runs Chi² → MUVR → FINAL FEATURES, then exits
```bash
maGeneLearn train \
  --no-split \
  --feature-selection-only \
  --train-meta test/skip_split/train_metadata.tsv \
  --features  test/skip_split/full_features.tsv \
  --name STEC \
  --label SYMP \
  --group-column t5 \
  --chisq --muvr \
  --muvr-model RFC \
  --k 5000 \
  --n-jobs -1 \
  --output-dir selected_features_split \
  
```


* **8.2) Skip Chi² (use an already-filtered matrix, still run MUVR)**  
In this scenario, you start with not-so-many features (Around 600K~1M). Therefore, you would like to skip Chi² step and use MUVR as the sole feature selection method. In this case, you will have to pass the same feature file into two different flags: `--features` and `--chisq-file`.

#### Runs split → MUVR → FINAL FEATURES, then exits

```bash
  maGeneLearn train \
  --feature-selection-only
  --meta-file test/skip_chi/2023_jp_meta_file.tsv \
  --chisq-file test/skip_chi/full_features.tsv \
  --features test/skip_chi/full_features.tsv \
  --name STEC \
  --muvr \
  --muvr-model RFC \
  --group-column t5 \
  --label SYMP \
  --lineage-col LINEAGE \
  --output-dir skip_chi_test
```

## 9 · Cite

* If you use maGeneLearn, please cite:

"Predicting clinical outcome of Escherichia coli O157:H7 infections using explainable Machine Learning"
https://doi.org/10.1101/2025.06.05.25329036 

"Optuna: A Next-generation Hyperparameter Optimization Framework"	
https://doi.org/10.48550/arXiv.1907.10902

See the following link to cite scikit-learn [https://scikit-learn.org/stable/about.html#citing-scikit-learn]

* If you use MUVR for feature selection:

"Variable selection and validation in multivariate modelling"
https://doi.org/10.1093/bioinformatics/bty710

* If you use feature interpretations with SHAP

"A Unified Approach to Interpreting Model Predictions"
https://doi.org/10.48550/arXiv.1705.07874

## 10 · Contact

Do you have any doubts? Please contact me at: j.a.paganini@uu.nl.


















import numpy as np
import sys
import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from py_muvr.feature_selector import FeatureSelector
from concurrent.futures import ProcessPoolExecutor
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging
from datetime import datetime

"""
02_feature_selection.py

Runs MUVR-based recursive or Boruta feature selection on a training dataset combined with a chisq feature matrix.

Inputs (CLI args):
  --train_data / -t          Path to input training data TSV. Must include:
                                * sample ID as first column (used as index)
                                * grouping column (specified via --group-col)
                                * outcome column (specified via --outcome-col)
  --chisq_file / -c          Path to full chisq feature matrix TSV. Must include:
                                * sample ID as first column (Index)
                                * all candidate feature columns (binary/integer values)
  --model / -m               Choice of classifier: 'RFC' (RandomForest) or 'XGBC' (XGBoost).
  --class_type / -y          Classification type (for encoding): 'binary' or 'multilabel'.
  --group-col / -g           Column name for grouping clusters (default: t5).
  --outcome-col / -u         Column name for outcome/label (default: SYMP).
  --filtered_train_dir / -f  Directory to write deduplicated training data TSV.
  --output / -o              Directory under which MUVR results will be saved.
  --name / -n                Base filename prefix for all outputs.
  --n-repetitions            Number of MUVR repetitions (default: 10)/Number of boruta repetitions (default: 100).
  --max-iter                 Number of boruta repetitions (default: 100).
  --n-outer                  Number of outer folds for MUVR (default: 5).
  --n-inner                  Number of inner folds for MUVR (default: 4).
  --metric                   Feature selection metric (default: MISS).
  --features-dropout-rate    Fraction of features to drop each iteration (default: 0.9).
  --remove_na                If NA or NaN values are found in the output variable, remove it.

Outputs:
  1. Filtered training data TSV:
       <filtered_train_dir>/<name>.tsv
     (samples deduplicated on ['group-col','outcome-col'], header preserved)

  2. MUVR-selected feature TSVs (three levels):
       <output>/<class_type>/<name>_<feature reduction>_<model>_min.tsv  # minimal feature set
       <output>/<class_type>/<name>_<feature reduction>_<model>_mid.tsv  # mid-level feature set
       <output>/<class_type>/<name>_<feature reduction>_<model>_max.tsv  # maximal feature set

Usage Example:
  python 02_feature_selection.py \
    --train_data data/train_set.tsv \
    --chisq_file data/full_chisq_matrix.tsv \
    --model RFC \
    --class_type binary \
    --group_col t5 \
    --outcome_col SYMP \
    --filtered_train_dir results/filtered_train \
    --output rsults/muvr_features \
    --name study1 \
    --n-repetitions 10 \
    --max-iter 100 \
    --n-outer 5 \
    --n-inner 4 \
    --metric MISS \
    --features-dropout-rate 0.9
    --remove_na
"""

def get_opts_muvr():
    parser = argparse.ArgumentParser(
        description="Run MUVR/Boruta-based feature selection on input data."
    )
    parser.add_argument('--train_data', '-t', type=str, required=True,
                        help='Path to training data TSV')
    parser.add_argument('--chisq_file', '-c', type=str, required=True,
                        help='Path to chisq features TSV')
    parser.add_argument('--model', '-m', type=str, choices=['RFC', 'XGBC'],
                        required=True, help='Model to use: RFC or XGBC')
    parser.add_argument('--group_col', '-g', type=str, default='t5',
                        help='Column name for grouping clusters (default: t5)')
    parser.add_argument('--outcome_col', '-u', type=str, default='SYMP',
                        help='Column name for outcome/label (default: SYMP)')
    parser.add_argument('--filtered_train_dir', '-f', type=str, required=True,
                        help='Directory to save the filtered training data (after deduplication)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for storing MUVR/Boruta results')
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='Base filename for outputs')
    parser.add_argument('--n-repetitions', type=int, default=10,
                        help='Number of MUVR repetitions (default: 10)')
    parser.add_argument('--max-iter', type=int, default=100,
                        help='Number of boruta repetitions (default: 100)')
    parser.add_argument('--n-outer', type=int, default=5,
                        help='Number of outer folds for MUVR (default: 5)')
    parser.add_argument('--n-inner', type=int, default=4,
                        help='Number of inner folds for MUVR (default: 4)')
    parser.add_argument('--metric', type=str, default='MISS',
                        help='Feature selection metric (default: MISS)')
    parser.add_argument('--features-dropout-rate', type=float, default=0.9,
                        help='Fraction of features to drop each iteration (default: 0.9)')
    parser.add_argument('--remove_na', action='store_true',
                        help = 'If set, drop any rows with NaN/NA in outcome or features (and warn)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs for MUVR/Boruta (default: 1 = sequential)')
    parser.add_argument('--method', type=str, choices=['muvr', 'boruta'], default='muvr', help='Feature selection method: muvr (default) or boruta')
    parser.add_argument('--perc', type=int, default=100, help='Boruta percentile for shadow feature comparison (default: 100)')
    parser.add_argument('--alpha', type=float, default=0.05, help='Boruta significance threshold (default: 0.05)')
    args = parser.parse_args()
    return (
        args.train_data,
        args.chisq_file,
        args.model,
        args.group_col,
        args.outcome_col,
        args.filtered_train_dir,
        args.output,
        args.name,
        args.n_repetitions,
        args.max_iter,
        args.n_outer,
        args.n_inner,
        args.metric,
        args.features_dropout_rate,
        args.remove_na,
        args.method,
        args.perc,
        args.alpha,
        args.n_jobs
    )

def setup_logging(output_dir: str, name: str, method: str, model: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobid = os.environ.get("SLURM_JOB_ID", "local")

    log_file = os.path.join(
        output_dir,
        f"{name}_{method}_{model}_{timestamp}_{jobid}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return log_file



def prepare_data_muvr(train_data, filtered_dir,name, group_col, outcome_col, remove_na=False):

    train_data_df = pd.read_csv(train_data, sep='\t', header=0, index_col=0)
    train_data_df.index = train_data_df.index.astype(str).str.strip()
    train_data_muvr = train_data_df.sort_index().drop_duplicates(subset=[group_col, outcome_col],
                                                       keep='last')

# 2) optionally drop missing outcomes early
    if remove_na:
        missing = train_data_muvr[outcome_col].isna()
        n_missing = missing.sum()
        if n_missing:
            logging.warning(
                "--remove_na: dropping %d rows with missing '%s'",
                n_missing,
                outcome_col
            )
            train_data_muvr = train_data_muvr.loc[~missing]

    # Ensure parent directory exists
    os.makedirs(filtered_dir, exist_ok=True)
    filtered_output_path = os.path.join(filtered_dir, f"{name}.tsv")

    train_data_muvr.to_csv(filtered_output_path, sep='\t')

    return train_data_muvr

def feature_reduction(train_data_muvr,chisq_file, model, output_dir,name, outcome_col, n_repetitions, max_iter, n_outer, n_inner, metric, features_dropout_rate, remove_na=False, n_jobs=1, method='muvr', perc=100, alpha=0.05):
    target_col = outcome_col
    train_data_muvr = train_data_muvr[[target_col]].copy()

    # Make sure metadata IDs are clean strings
    train_data_muvr.index = (
        train_data_muvr.index
        .astype(str)
        .str.strip()
    )

    logging.info("Loading Chi² features")

    # ------------------------------------------------------------------
    # Read header first so that we can explicitly set only feature
    # columns to int8 while leaving the sample-ID column as a string.
    # ------------------------------------------------------------------
    with open(chisq_file, "r") as f:
        header = f.readline().rstrip("\n\r").split("\t")

    if len(header) < 2:
        raise ValueError(
            "Chi² feature file must contain a sample-ID column "
            "and at least one feature column."
        )

    id_col = header[0]
    feature_names = header[1:]

    logging.info(
        "Chi² matrix contains %d feature columns",
        len(feature_names)
    )

    # Explicit int8 dtype prevents pandas from first constructing a much
    # larger int64 matrix.
    feature_dtypes = {
        feature: "int8"
        for feature in feature_names
    }

    chisq_df = pd.read_csv(
        chisq_file,
        sep="\t",
        header=0,
        index_col=0,
        dtype=feature_dtypes,
    )

    chisq_df.index = (
        chisq_df.index
        .astype(str)
        .str.strip()
    )

    if not chisq_df.index.is_unique:
        raise ValueError(
            "Duplicate sample IDs found in Chi² feature matrix."
        )

    logging.info(
        "Chi² matrix loaded: %d isolates x %d features",
        chisq_df.shape[0],
        chisq_df.shape[1]
    )

    # ------------------------------------------------------------------
    # Keep only samples present in BOTH metadata and Chi² matrix.
    #
    # Preserve the row order of the Chi² file, matching the behavior of
    # the previous chunksize=1 implementation.
    # ------------------------------------------------------------------
    metadata_ids = set(train_data_muvr.index)

    common_ids = [
        sample_id
        for sample_id in chisq_df.index
        if sample_id in metadata_ids
    ]

    if len(common_ids) == 0:
        raise ValueError(
            "No overlapping sample IDs between training metadata "
            "and Chi² feature matrix."
        )

    logging.info(
        "Samples shared between metadata and Chi² matrix: %d",
        len(common_ids)
    )

    # Align both objects once
    train_labels = train_data_muvr.loc[common_ids]
    chisq_df = chisq_df.loc[common_ids]

    # One concatenation instead of one concatenation per isolate
    model_input = pd.concat(
        [
            train_labels,
            chisq_df
        ],
        axis=1
    )

    logging.info(
        "Model input assembled: %d isolates x %d features",
        model_input.shape[0],
        model_input.shape[1] - 1
    )

    # We no longer need the standalone Chi² dataframe
    del chisq_df
    del train_labels
    del feature_dtypes

    if remove_na:
        # check for NaNs in outcome
        missing_labels = model_input[outcome_col].isna()
        count_labels = missing_labels.sum()
        # check for NaNs anywhere in feature matrix
        features = model_input.drop(columns=[outcome_col])
        missing_features = features.isna().any(axis=1)
        count_feats = missing_features.sum()

        total_to_drop = (missing_labels | missing_features).sum()
        if total_to_drop > 0:
            logging.warning(
                "--remove_na set: dropping %d rows (%d missing labels, %d missing features)",
                total_to_drop,
                count_labels,
                count_feats
            )

            # drop them
            model_input = model_input.loc[~(missing_labels | missing_features)]

    y_series = model_input[target_col]

    if model=='XGBC':
        y_encoded = LabelEncoder().fit_transform(y_series)
        y_variable = y_encoded
    elif model=='RFC':
        y_variable = y_series.values.ravel()

    else:
        sys.exit("Select a valid model: RFC or XGBC")

    X_muvr = model_input.drop(columns=[target_col]).to_numpy()
    feature_names = model_input.drop(columns=[target_col]).columns

    logging.info(
        "Feature-selection input: %d isolates and %d features",
        model_input.shape[0],
        len(feature_names)
    )

    if method == "muvr":
        feature_selector = FeatureSelector(
            n_repetitions=n_repetitions,
            n_outer=n_outer,
            n_inner=n_inner,
            estimator=model,
            metric=metric,
            features_dropout_rate=features_dropout_rate
        )

        logging.info("Running MUVR")

        executor = None
        if n_jobs != 1:
            executor = ProcessPoolExecutor(max_workers=n_jobs)

        feature_selector.fit(X_muvr, y_variable, executor=executor)
        selected_features = feature_selector.get_selected_features(feature_names=feature_names)

        logging.info(
            "Number of MUVR selected features - Min: %d, Mid: %d, Max: %d",
            len(selected_features.min),
            len(selected_features.mid),
            len(selected_features.max)
        )

        # Obtain a dataframe containing MUVR selected features
        df_min = model_input[list(selected_features.min)]
        df_mid = model_input[list(selected_features.mid)]
        df_max = model_input[list(selected_features.max)]

        #Write features to a new file.
        os.makedirs(output_dir, exist_ok=True)
        min_features_file_name = os.path.join(output_dir, f'{name}_muvr_{model}_min.tsv')
        mid_features_file_name = os.path.join(output_dir, f'{name}_muvr_{model}_mid.tsv')
        max_features_file_name = os.path.join(output_dir, f'{name}_muvr_{model}_max.tsv')

        df_min.to_csv(min_features_file_name, sep='\t')
        df_mid.to_csv(mid_features_file_name, sep='\t')
        df_max.to_csv(max_features_file_name, sep='\t')

        return df_min,df_mid,df_max
    
    elif method == "boruta":
        logging.info("Running Boruta feature selection")

        if model == "RFC":
            estimator = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced", random_state=42)
        elif model == "XGBC":
            estimator = XGBClassifier(n_jobs=n_jobs, eval_metric="logloss", random_state=42)
        else:
            sys.exit("Boruta only supports RFC or XGBC")
        
        boruta = BorutaPy(
            estimator=estimator,
            n_estimators='auto',
            verbose=2,
            random_state=42,
            max_iter=max_iter,
            perc=perc,
            alpha=alpha
        )

        boruta.fit(X_muvr, y_variable)

        selected_mask = boruta.support_
        selected_features = feature_names[selected_mask]

        if len(selected_features) == 0:
            sys.exit("Boruta rejected all features. Consider increasing max_iter, lowering perc, or increasing alpha.")
        
        df_min = model_input[list(selected_features)]

        os.makedirs(output_dir, exist_ok=True)
        min_features_file_name = os.path.join(output_dir, f'{name}_boruta_{model}_min.tsv')

        df_min.to_csv(min_features_file_name, sep='\t')

        return df_min

#######################################################
#
#                  MAIN                               #
#
#######################################################
if __name__ == "__main__":
    if __name__ == "__main__":
        (
            train_data,
            chisq_file,
            model,
            group_col,
            outcome_col,
            filtered_train_dir,
            output_dir,
            name,
            n_repetitions,
            max_iter,
            n_outer,
            n_inner,
            metric,
            features_dropout_rate,
            remove_na,
            method,
            perc,
            alpha,
            n_jobs
        ) = get_opts_muvr()

        log_file = setup_logging(output_dir, name, method, model)
        logging.info("Writing feature-selection log to %s", log_file)

        logging.info("Filtering data")

        train_filtered = prepare_data_muvr(
            train_data,
            filtered_train_dir,
            name,
            group_col,
            outcome_col,
            remove_na=remove_na
        )

        logging.info("Running MUVR/Boruta feature reduction")
        feature_reduction(
            train_filtered,
            chisq_file,
            model,
            output_dir,
            name,
            outcome_col,
            n_repetitions,
            max_iter,
            n_outer,
            n_inner,
            metric,
            features_dropout_rate,
            remove_na=remove_na,
            method=method,
            perc=perc,
            alpha=alpha,
            n_jobs=n_jobs
        )


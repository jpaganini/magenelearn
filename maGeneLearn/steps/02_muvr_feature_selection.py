import numpy as np
import sys
import os
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from py_muvr.feature_selector import FeatureSelector

"""
02_muvr_feature_selection.py

Runs MUVR-based recursive feature selection on a training dataset combined with a chisq feature matrix.

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
  --n-repetitions            Number of MUVR repetitions (default: 10).
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
       <output>/<class_type>/<name>_muvr_<model>_min.tsv  # minimal feature set
       <output>/<class_type>/<name>_muvr_<model>_mid.tsv  # mid-level feature set
       <output>/<class_type>/<name>_muvr_<model>_max.tsv  # maximal feature set

Usage Example:
  python 02_muvr_feature_selection.py \
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
    --n-outer 5 \
    --n-inner 4 \
    --metric MISS \
    --features-dropout-rate 0.9
    --remove_na
"""

def get_opts_muvr():
    parser = argparse.ArgumentParser(
        description="Run MUVR-based feature selection on input data."
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
                        help='Output directory for storing MUVR results')
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='Base filename for outputs')
    parser.add_argument('--n-repetitions', type=int, default=10,
                        help='Number of MUVR repetitions (default: 10)')
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
        args.n_outer,
        args.n_inner,
        args.metric,
        args.features_dropout_rate,
        args.remove_na
    )

def prepare_data_muvr(train_data, filtered_dir,name, group_col, outcome_col, remove_na=False):

    train_data_df = pd.read_csv(train_data, sep='\t', header=0, index_col=0)

    train_data_muvr = train_data_df.sort_index().drop_duplicates(subset=[group_col, outcome_col],
                                                       keep='last')

# 2) optionally drop missing outcomes early
    if remove_na:
        missing = train_data_muvr[outcome_col].isna()
        n_missing = missing.sum()
        if n_missing:
            print(f"WARNING: --remove_na: dropping {n_missing} rows with missing '{outcome_col}'")
            train_data_muvr = train_data_muvr.loc[~missing]

    # Ensure parent directory exists
    os.makedirs(filtered_dir, exist_ok=True)
    filtered_output_path = os.path.join(filtered_dir, f"{name}.tsv")

    train_data_muvr.to_csv(filtered_output_path, sep='\t')

    return train_data_muvr

def feature_reduction(train_data_muvr,chisq_file, model, output_dir,name, outcome_col, n_repetitions, n_outer, n_inner, metric, features_dropout_rate, remove_na=False):

    target_col = outcome_col
    train_data_muvr = train_data_muvr[[target_col]]

    # Create an iterator for reading chisq_features line by line
    reader_chisq = pd.read_csv(chisq_file, sep='\t', header=0, iterator=True, chunksize=1)

    # Create a dataframe to hold the results
    model_input = pd.DataFrame()

    print("Loading chisq feateres")
    # Get the first line of chisq_features
    try:
        chunk_chisq = next(reader_chisq)
    except StopIteration:
        chunk_chisq = pd.DataFrame()

    while not chunk_chisq.empty:
        # Set the index as the first column
        chunk_chisq.set_index(chunk_chisq.columns[0], inplace=True)
        chunk_chisq = chunk_chisq.astype("int8")

        # Merge the current line with isolate_metadata based on your desired criteria
        merged_line = pd.merge(train_data_muvr, chunk_chisq, left_index=True, right_index=True, how='inner')
        #print(merged_line)
        model_input = pd.concat([model_input, merged_line], ignore_index=False)

        #Get the following lines of the dataframe
        try:
            chunk_chisq = next(reader_chisq)
        except StopIteration:
            chunk_chisq = pd.DataFrame()

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
            print(f"WARNING: --remove_na set: dropping {total_to_drop} rows "
                  f"({count_labels} missing labels, {count_feats} missing features)")

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

    feature_selector = FeatureSelector(
        n_repetitions=n_repetitions,
        n_outer=n_outer,
        n_inner=n_inner,
        estimator=model,
        metric=metric,
        features_dropout_rate=features_dropout_rate
    )

    print("Running MUVR")
    feature_selector.fit(X_muvr, y_variable)
    selected_features = feature_selector.get_selected_features(feature_names=feature_names)

    # Obtain a dataframe containing MUVR selected features
    df_muvr_min = model_input[[outcome_col]+list(selected_features.min)]
    df_muvr_mid = model_input[[outcome_col]+list(selected_features.mid)]
    df_muvr_max = model_input[[outcome_col]+list(selected_features.max)]

    #Write features to a new file.
    os.makedirs(output_dir, exist_ok=True)
    min_features_file_name = os.path.join(output_dir, f'{name}_muvr_{model}_min.tsv')
    mid_features_file_name = os.path.join(output_dir, f'{name}_muvr_{model}_mid.tsv')
    max_features_file_name = os.path.join(output_dir, f'{name}_muvr_{model}_max.tsv')

    df_muvr_min.to_csv(min_features_file_name, sep='\t')
    df_muvr_mid.to_csv(mid_features_file_name, sep='\t')
    df_muvr_max.to_csv(max_features_file_name, sep='\t')

    return df_muvr_min,df_muvr_mid,df_muvr_max

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
            n_outer,
            n_inner,
            metric,
            features_dropout_rate,
            remove_na
        ) = get_opts_muvr()
        print("Filtering data")
        train_filtered = prepare_data_muvr(
            train_data,
            filtered_train_dir,
            name,
            group_col,
            outcome_col,
            remove_na=remove_na
        )

        print("Running MUVR feature reduction")
        feature_reduction(
            train_filtered,
            chisq_file,
            model,
            output_dir,
            name,
            outcome_col,
            n_repetitions,
            n_outer,
            n_inner,
            metric,
            features_dropout_rate,
            remove_na=remove_na
        )



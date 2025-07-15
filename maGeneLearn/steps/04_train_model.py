"""
04_train_model.py
Train and tune a RandomForest or XGBoost classifier with grouped cross-validation.

Usage:
    python train_model.py
        --features PATH           Path to TSV file containing features, label, and group column (index in first column)
        --label LABEL_COL         Name of the column to use as the target label
        --model {RFC,XGBC}        Which model to train: RFC (RandomForestClassifier) or XGBC (XGBClassifier)
        --sampling {none,random,smote}
                                  Oversampling strategy: none (no oversampling), random (RandomOverSampler), or smote (SMOTE)
        --group_column GROUP_COL  Column name in the TSV that contains group IDs for cross-validation
        --output_model DIR        Directory to save the trained model file
        --output_cv DIR           Directory to save the CV results file
        --name BASE_NAME          Base name for outputs; model -> DIR/BASE_NAME.joblib,
                                  CV results -> DIR/BASE_NAME_cv.tsv
        [--n_iter N]              Number of hyperparameter settings to sample (default: 100)
        [--scoring METRIC [METRIC ...]]
                                  One or more scoring metrics for evaluation and refit (default: balanced_accuracy)
        [--n_splits K]            Number of CV folds (default: 5)

Example:
    python 04_train_model.py \
      --features data/features.tsv \
      --label target \
      --model XGBC \
      --sampling smote \
      --group_column batch_id \
      --output_model models/ \
      --output_cv results/ \
      --name experiment1 \
      --n_iter 50 \
      --scoring balanced_accuracy f1 \
      --n_splits 5

Inputs:
    • A tab-separated values (TSV) file whose first column is the sample ID,
      followed by feature columns, one column for the label, and one for group IDs.
    • Command-line options as above.

Outputs:
    • A serialized model file saved as: {output_model}/{BASE_NAME}.joblib
    • A TSV file of the RandomizedSearchCV cv_results_ saved as: {output_cv}/{BASE_NAME}_cv.tsv

"""

# --- Standard libraries ---
import argparse             # For command-line argument parsing
import os                   # For file path operations
import sys                  # For exiting on errors
from pathlib import Path

# --- Scientific computing ---
import numpy as np         # Numerical arrays
import pandas as pd        # DataFrame operations
from sklearn.utils.class_weight import compute_sample_weight  # For weighted training
from scipy.sparse import issparse

# --- Machine Learning ---
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline

# --- Model saving ---
import joblib               # For model serialization
from sklearn.preprocessing import LabelEncoder # For encoding string labels

# Set a global random seed for reproducibility
RSEED = 50

# Configuration of supported models and their hyperparameter search spaces
# === common lists -------------------------------------------------
N_ESTIMATORS = [int(x) for x in np.linspace(200, 1000, 10)]
MAX_DEPTH_RF  = [int(x) for x in np.linspace(100, 500, 11)]
MAX_FEATS_RF  = ['log2', 'sqrt']

# SMOTE neighbours
SMOTE_K = [1, 2, 3, 4]

# XGBoost space
ETA          = np.linspace(0.01, 0.2, 10).tolist()
GAMMA        = [0, 3, 5, 7, 9]
MAX_DEPTH_XG = [3, 4, 5, 6, 7, 8, 9, 10]
MIN_CHILD_W  = [1, 2, 3, 4, 5]
SUBSAMPLE    = [0.6, 0.7, 0.8, 0.9, 1]
COLSAMPLE    = [0.7, 0.8, 0.9, 1]

MODEL_PARAMS = {
    "RFC": {
        "estimator": RandomForestClassifier(random_state=RSEED, n_jobs=-1),
        "param_grid": {
            "model__n_estimators": N_ESTIMATORS,
            "model__max_depth":    MAX_DEPTH_RF,
            "model__max_features": MAX_FEATS_RF,
        },
    },
    "XGBC": {
        "estimator": XGBClassifier(
            random_state=RSEED,
            n_jobs=1,                 # let joblib handle outer parallelism
            use_label_encoder=False,
            eval_metric="logloss",
        ),
        "param_grid": {
            "model__n_estimators":     N_ESTIMATORS,
            "model__learning_rate":    ETA,
            "model__gamma":            GAMMA,
            "model__max_depth":        MAX_DEPTH_XG,
            "model__min_child_weight": MIN_CHILD_W,
            "model__subsample":        SUBSAMPLE,
            "model__colsample_bytree": COLSAMPLE,
        },
    },
}


def parse_arguments():
    """
    Parse and validate command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Train and tune ML model with grouped cross-validation."
    )
    parser.add_argument('--features', required=True,
                        help='Path to TSV file containing features + label + group column')
    parser.add_argument('--label', required=True,
                        help='Which column to use as the target label')
    parser.add_argument('--model', choices=list(MODEL_PARAMS), required=True,
                        help='Which model to train: RFC or XGBC')
    parser.add_argument('--sampling', choices=['none','random','smote'], default='none',
                        help='Oversampling strategy: none, random, or smote')
    parser.add_argument('--group_column', required=True,
                        help='Column name in the input file that contains group IDs for CV')
    parser.add_argument('--name', required=True,
                        help='Base name for outputs; model -> NAME.joblib, CV -> NAME_cv.tsv')
    parser.add_argument('--output_model', required=True,
                        help='File path to save the trained model (.joblib)')
    parser.add_argument('--output_cv', required=True,
                        help='File path to save CV results as TSV')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of parameter settings sampled in RandomizedSearchCV (default: 100)')
    parser.add_argument('--scoring', nargs='+', default=['balanced_accuracy'],
                        help='One or more scoring metrics for evaluation and refit (default: balanced_accuracy)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    return parser.parse_args()


def load_data(path, label_col, group_col):
    """
    Load feature DataFrame and extract target and grouping.
    Args:
        path (str): Path to TSV file (features + label).
        label_col (str): Column name to use as label.
    Returns:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Label vector.
        groups (np.ndarray): Group IDs for CV.
    """
    df = pd.read_csv(path, sep='\t', index_col=0)
    if label_col not in df.columns:
        sys.exit(f"Error: label column '{label_col}' not found in input.")
    y = df[label_col].values
    # Determine grouping column for cross-validation
    if group_col not in df.columns:
        sys.exit(f"Error: Group column '{group_col}' not found in input.")
    groups = df[group_col].values
    X = df.drop(columns=[label_col,group_col])

    return X, y, groups


def prepare_pipeline(model_key, sampling):
    """
    Construct an imblearn Pipeline with optional oversampling and the chosen estimator.
    Args:
        model_key (str): Key in MODEL_PARAMS ('RFC' or 'XGBC').
        sampling (str): 'none', 'random', or 'smote'.
    Returns:
        pipeline (Pipeline): Configured pipeline object.
        param_grid (dict): Hyperparameter grid for RandomizedSearchCV.
    """
    cfg = MODEL_PARAMS[model_key]
    steps, grid = [], cfg["param_grid"].copy()

    if sampling == "random":
        steps.append(("oversampler", RandomOverSampler(random_state=RSEED)))
    elif sampling == "smote":
        steps.append(("oversampler", SMOTE(random_state=RSEED)))
        # expose SMOTE hyper-parameter
        grid["oversampler__k_neighbors"] = SMOTE_K

    steps.append(("model", cfg["estimator"]))
    return Pipeline(steps), grid


def get_cv_splits(X, y, groups,n_splits):
    """
    Create cross-validation splits based on grouping strategy.
    Creates StratifiedGroupKFold
    Returns:
        list: List of (train_idx, test_idx) splits.
    """
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RSEED
    )
    return list(cv.split(X, y, groups))




def search_hyperparameters(pipeline, param_grid, cv_splits, X, y,
                           model_key, sampling, n_iter, scoring):
    """
    Perform RandomizedSearchCV over the pipeline and parameter grid.
    Args:
        pipeline (Pipeline): Pipeline to optimize.
        param_grid (dict): Hyperparameter grid.
        cv_splits (list): Precomputed CV splits.
        n_iter (int): Number of parameter settings to sample.
        scoring (list): Scoring metrics for evaluation and refit.
    Returns:
        RandomizedSearchCV: Fitted search object.
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        refit=scoring[0],  # refit on first scoring metric by default
        cv=cv_splits,
        random_state=RSEED,
        verbose=2,
        n_jobs=-1
    )
    # If XGBC without oversampling, supply sample weights
    if model_key == 'XGBC' and sampling == 'none':
        sw = compute_sample_weight('balanced', y)
        search.fit(X, y, model__sample_weight=sw)
    else:
        search.fit(X, y)
    return search


def save_model_and_cv(search, model_dir, cv_dir, name, model, sampling):
    """
    Save the trained model and cross-validation results to disk.
    Args:
        search (RandomizedSearchCV): Fitted search object.
        model_path (str): File path to save best model (.joblib).
        cv_path (str): File path to save CV results (.tsv).
    """
    # Create parent directories only if a directory component exists
    Path(model_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    Path(cv_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(model_dir, f"{name}_{model}_{sampling}.joblib")
    cv_path    = os.path.join(cv_dir, f"CV_{name}_{model}_{sampling}.tsv")

    joblib.dump(search.best_estimator_, model_path)
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df.to_csv(cv_path, sep='\t', index=False)


def main():
    """
    Main execution flow:
      - Parse arguments
      - Load data
      - Build pipeline
      - Define CV
      - Run hyperparameter search
      - Save model and CV results
      - Optionally evaluate and save report
    """
    args = parse_arguments()
    X, y, groups = load_data(args.features, args.label, args.group_column)
    if args.model == 'XGBC':
        le = LabelEncoder()
        y = le.fit_transform(y)

    if args.sampling == 'smote' and issparse(X):
        sys.exit("Error: SMOTE cannot be applied to a sparse matrix. "
                 "Choose --sampling random or none, or densify the data first.")
    pipeline, param_grid = prepare_pipeline(args.model, args.sampling)
    cv_splits = get_cv_splits(X, y, groups, args.n_splits)

    print("Starting hyperparameter search...")
    search = search_hyperparameters(
        pipeline, param_grid, cv_splits,
        X, y, args.model, args.sampling,args.n_iter, args.scoring)

    save_model_and_cv(search, args.output_model, args.output_cv, args.name, args.model, args.sampling)
    print(f"Training and CV complete. Model saved in '{args.output_model}' and CV results in '{args.output_cv}'.")

    print("Training and CV complete. Model and results saved.")


if __name__ == '__main__':
    main()

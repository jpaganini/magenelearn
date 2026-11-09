"""
04_train_model.py
Train and tune a RandomForest or XGBoost classifier with grouped cross-validation.

Usage:
    python train_model.py
        --features PATH           Path to TSV file containing features, label, and group column (index in first column)
        --label LABEL_COL         Name of the column to use as the target label
        --model {RFC,XGBC, SVM}        Which model to train: RFC (RandomForestClassifier), XGBC (XGBClassifier) or SVM
        --sampling {none,random,smote,smoteenn,enn,random_under}
                                  Oversampling strategy: none (no oversampling), random (RandomOverSampler), smote (SMOTE), (RandomUnderSampler), smoteenn (SMOTEENN), or enn (EditedNearestNeighbours
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
import time
import logging
from datetime import datetime

# --- Machine Learning ---
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# --- Model saving ---
import joblib               # For model serialization
from sklearn.preprocessing import LabelEncoder # For encoding string labels
import optuna
from sklearn.model_selection import cross_validate

# Set a global random seed for reproducibility
RSEED = 50

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
    parser.add_argument('--model', choices=['RFC', 'XGBC', 'SVM', 'LR'], required=True,
                        help='Which model to train: RFC, XGBC or SVM')
    parser.add_argument('--sampling', choices=['none','random','smote', 'random_under', 'enn', 'smoteenn'], default='none',
                        help='Oversampling strategy: none, random, smote, enn, random_under or smoteenn')
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
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for CV and model training (default: -1 = all cores)')
    parser.add_argument('--lr-penalty', choices=['l1', 'l2', 'elasticnet'], default='l2',
                        help='Penalty type for Logistic Regression (default: l2)')
    parser.add_argument('--xgb-policy', choices=['depthwise', 'lossguide'], default='depthwise',
                        help='Tree growth policy for XGBoost (default: depthwise)')
    return parser.parse_args()

def setup_logging(output_dir, name, model, sampling):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"{name}_{model}_{sampling}_{timestamp}.log"

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

def load_data(path, label_col, group_col):
    df = pd.read_csv(path, sep='\t', index_col=0)
    if label_col not in df.columns:
        sys.exit(f"Error: label column '{label_col}' not found in input.")
    if group_col not in df.columns:
        sys.exit(f"Error: group column '{group_col}' not found in input.")
    y = df[label_col].values
    groups = df[group_col].values
    X = df.drop(columns=[label_col, group_col])
    return X, y, groups


def prepare_pipeline(model_key, sampling, n_jobs, lr_penalty="l2",xgb_policy="depthwise"):
    steps = []
    if sampling == "random":
        steps.append(("sampler", RandomOverSampler(random_state=RSEED)))
    elif sampling == "smote":
        steps.append(("sampler", SMOTE(random_state=RSEED)))  # k_neighbors tuned in Optuna
    elif sampling == "random_under":
        steps.append(("sampler", RandomUnderSampler(random_state=RSEED))) #random undersampling
    elif sampling == "enn":
        steps.append(("sampler", EditedNearestNeighbours(kind_sel="mode"))) #n_neighbors tuned in Optuna
    elif sampling == "smoteenn":
        steps.append(("sampler", SMOTEENN(smote=SMOTE(random_state=RSEED), enn=EditedNearestNeighbours(kind_sel="mode"), random_state=RSEED))) #k_neighbors and n_neighbors tuned in Optuna


    if model_key == "RFC":
        estimator = RandomForestClassifier(random_state=RSEED, n_jobs=n_jobs)
    elif model_key == "XGBC":
        if xgb_policy == "lossguide":
            estimator = XGBClassifier(
                random_state=RSEED,
                n_jobs=n_jobs,
                tree_method="hist",
                grow_policy="lossguide",
                max_depth=0,  # required when using max_leaves
                use_label_encoder=False,
                eval_metric="logloss",
            )
        else:
            estimator = XGBClassifier(
                random_state=RSEED,
                n_jobs=n_jobs,  # let joblib handle outer CV parallelism
                use_label_encoder=False,
                eval_metric="logloss",
            )

    elif model_key == "SVM":
        estimator = SVC(random_state=RSEED, probability=True)  # probability=True for ROC/AUC support


    elif model_key == "LR":
        estimator = LogisticRegression(
            max_iter=5000,
            solver="saga",
            multi_class="multinomial",
            penalty=lr_penalty,
            random_state=RSEED,
            n_jobs=n_jobs
        )

    steps.append(("model", estimator))

    return Pipeline(steps)


def get_cv_splits(X, y, groups, n_splits):
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RSEED
    )
    return list(cv.split(X, y, groups))


def optuna_objective(trial, pipeline, X, y, groups, cv_splits, scoring, model_key, sampling,n_jobs):
    model = clone(pipeline)
    if model_key == "XGBC":
        if hasattr(model.named_steps["model"], "grow_policy") and model.named_steps["model"].grow_policy == "lossguide":
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 2000, step=200),
                "model__learning_rate": trial.suggest_float("model__learning_rate", 0.005, 0.2, log=True),
                "model__max_leaves": trial.suggest_int("model__max_leaves", 16, 64, step=8),
                "model__min_child_weight": trial.suggest_int("model__min_child_weight", 1, 50),
                "model__gamma": trial.suggest_float("model__gamma", 0, 5),
                "model__subsample": trial.suggest_float("model__subsample", 0.5, 1.0),
                "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
                "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
                "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True),
            }
        else:
            params = {
                "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 2000, step=200),
                "model__learning_rate": trial.suggest_float("model__learning_rate", 0.005, 0.2, log=True),
                "model__max_depth": trial.suggest_int("model__max_depth", 3, 7),
                "model__min_child_weight": trial.suggest_int("model__min_child_weight", 1, 50),
                "model__gamma": trial.suggest_float("model__gamma", 0, 5),
                "model__subsample": trial.suggest_float("model__subsample", 0.5, 1.0),
                "model__colsample_bytree": trial.suggest_float("model__colsample_bytree", 0.5, 1.0),
                "model__reg_alpha": trial.suggest_float("model__reg_alpha", 1e-8, 10.0, log=True),
                "model__reg_lambda": trial.suggest_float("model__reg_lambda", 1e-8, 10.0, log=True),
            }
    elif model_key == "RFC":  # RFC
        params = {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 200, 1500, step=100),
            "model__max_depth": trial.suggest_int("model__max_depth", 10, 100, step=10),
            "model__max_features": trial.suggest_categorical("model__max_features", ["log2", "sqrt", 0.2, 0.5])
        }
        if sampling == "none":
            params["model__class_weight"] = trial.suggest_categorical("model__class_weight", [None, "balanced"])

    elif model_key == "SVM":
        params = {
            "model__C": trial.suggest_loguniform("model__C", 1e-3, 1e3),
            "model__kernel": trial.suggest_categorical("model__kernel", ["linear", "rbf"]),
            "model__gamma": trial.suggest_categorical("model__gamma", ["scale", "auto"]),
            "model__probability": False,
        }
        if sampling == "none":
            params["model__class_weight"] = trial.suggest_categorical("model__class_weight", [None, "balanced"])

    elif model_key == "LR":
        params = {
            "model__C": trial.suggest_float("model__C", 1e-3, 1e3, log=True),
        }
        if model.named_steps["model"].penalty == "elasticnet":
            params["model__l1_ratio"] = trial.suggest_float("model__l1_ratio", 0.0, 1.0)
        if sampling == "none":
            params["model__class_weight"] = trial.suggest_categorical(
                "model__class_weight", [None, "balanced"]
            )

    if sampling == "smote":
        params["sampler__k_neighbors"] = trial.suggest_int("sampler__k_neighbors", 3, 10)
    
    elif sampling == "enn":
        params["sampler__n_neighbors"] = trial.suggest_int("sampler__n_neighbors", 3, 10)
    
    elif sampling == "smoteenn":
        params["sampler__smote__k_neighbors"] = trial.suggest_int("sampler__smote__k_neighbors", 3, 10)
        params["sampler__enn__n_neighbors"] = trial.suggest_int("sampler__enn__n_neighbors", 3, 10)



    model.set_params(**params)

    start = time.time()

    if model_key == 'XGBC' and sampling == 'none':
        sw = compute_sample_weight("balanced", y)
        cv_results = cross_validate(
            model, X, y, groups=groups, cv=cv_splits,
            scoring=scoring, fit_params={"model__sample_weight": sw}, n_jobs=n_jobs
        )
    else:
        cv_results = cross_validate(
            model, X, y, groups=groups, cv=cv_splits,
            scoring=scoring, n_jobs=n_jobs
        )

    elapsed = time.time() - start

    metric = scoring[0]
    mean_score = np.mean(cv_results[f"test_{metric}"])
    std_score = np.std(cv_results[f"test_{metric}"])
    scores = cv_results[f"test_{metric}"]

    logging.info(
        f"[Trial {trial.number}] {model_key} | "
        f"Mean {metric}: {mean_score:.4f} "
        f"(±{std_score:.4f}) | "
        f"Time: {elapsed:.1f}s"
    )


    trial_data = {
        "trial": trial.number,
        "score_mean": np.mean(scores),
        "score_std": np.std(scores),
        **{f"fold{i}_score": s for i, s in enumerate(scores)},
        **params,
        "elapsed_time": elapsed
    }

    return np.mean(scores), trial_data



def search_hyperparameters_optuna(pipeline, X, y, groups, cv_splits,
                                  model_key, sampling, n_iter, scoring,n_jobs):
    trial_history = []

    def _objective(trial):
        score, trial_data = optuna_objective(trial, pipeline, X, y, groups, cv_splits, scoring, model_key, sampling,n_jobs=1)
        trial_history.append(trial_data)
        
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RSEED))
    study.optimize(_objective, n_trials=n_iter, n_jobs=n_jobs)

    best_params = study.best_params
    best_score = study.best_value

    final_model = clone(pipeline)
    final_model.set_params(**best_params)

    if model_key == "XGBC" and sampling == "none":
        sw = compute_sample_weight("balanced", y)
        final_model.fit(X, y, model__sample_weight=sw)
    else:
        final_model.fit(X, y)


    return final_model, pd.DataFrame(trial_history), best_params, best_score


def main():
    args = parse_arguments()

    log_file = setup_logging(
        args.output_model,
        args.name,
        args.model,
        args.sampling
    )

    logging.info("Writing training log to %s", log_file)

    X, y, groups = load_data(args.features, args.label, args.group_column)

    logging.info(
        "Loaded dataset: %d isolates, %d features",
        X.shape[0],
        X.shape[1]
    )
    logging.info("Classes: %s", list(np.unique(y)))
    logging.info("Groups: %d unique", len(np.unique(groups)))

    if args.sampling in ['smote', 'enn', 'smoteenn'] or args.model == 'XGBC':
        le = LabelEncoder()
        y = le.fit_transform(y)

    if args.sampling in ['smote', 'smoteenn', 'enn'] and issparse(X):
        sys.exit(f"Error: {args.sampling.upper()} cannot be applied to a sparse matrix. Densify first.")

    pipeline = prepare_pipeline(
        args.model,
        args.sampling,
        args.n_jobs,
        lr_penalty=args.lr_penalty,
        xgb_policy=args.xgb_policy
    )

    logging.info("Model: %s | Sampling: %s", args.model, args.sampling)
    logging.info("Pipeline:\n%s", pipeline)

    cv_splits = get_cv_splits(X, y, groups, args.n_splits)

    logging.info("Prepared %d CV folds", len(cv_splits))
    logging.info("Starting Optuna hyperparameter optimization...")

    best_model, trials_df, best_params, best_score = search_hyperparameters_optuna(
        pipeline, X, y, groups, cv_splits,
        args.model, args.sampling, args.n_iter, args.scoring, args.n_jobs
    )

    Path(args.output_model).mkdir(parents=True, exist_ok=True)
    Path(args.output_cv).mkdir(parents=True, exist_ok=True)

    if args.model == "LR":
        model_path = os.path.join(
            args.output_model,
            f"{args.name}_{args.model}_{args.sampling}_{args.lr_penalty}.joblib"
        )
        cv_path = os.path.join(
            args.output_cv,
            f"CV_{args.name}_{args.model}_{args.sampling}_{args.lr_penalty}.tsv"
        )
    else:
        model_path = os.path.join(
            args.output_model,
            f"{args.name}_{args.model}_{args.sampling}.joblib"
        )
        cv_path = os.path.join(
            args.output_cv,
            f"CV_{args.name}_{args.model}_{args.sampling}.tsv"
        )

    if args.sampling in ['smote', 'enn', 'smoteenn'] or args.model == "XGBC":
        joblib.dump({"model": best_model, "label_encoder": le}, model_path)
    else:
        joblib.dump(best_model, model_path)

    trials_df.to_csv(cv_path, sep="\t", index=False)

    logging.info("Best params: %s", best_params)
    logging.info("Best score (%s): %.4f", args.scoring[0], best_score)
    logging.info(
        "Model saved in '%s' and CV results in '%s'.",
        model_path,
        cv_path
    )


if __name__ == '__main__':
    main()

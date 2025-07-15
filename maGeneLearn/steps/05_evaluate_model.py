#!/usr/bin/env python
"""
Script: 05_evaluate_model.py

Inputs:
  --model          Path to trained .joblib model artifact.
  --features       Path to TSV file containing features, label, and group columns.
  --label          Name of the label column in the TSV.
  --group_column   Name of the group column in the TSV.
  --n_splits       Number of cross-validation folds (mutually exclusive with --no_cv). Useful for evaluating performance on the TRAINING set.
  --no_cv          If set, skip cross-validation and predict on the full dataset once. Useful for evaluating performance on TEST split.
  --output_dir     Directory in which to write outputs (will be created if it does not exist).
  --name           Prefix to add to all output filenames.
  --log_level      Logging verbosity level (choices: DEBUG, INFO, WARNING, ERROR). Default: INFO.
  --fasta          If set, it produces a fasta-formatted output of the features used to train the model. Features are named according to importance based on the feature_importance() function from scikit-learn.
                   Works if features are DNA sequences, and if sequences match column names.
  --predict_only   If set, it predicts new samples that don't contain labels. So, no evaluation metrics are calculated.

Outputs:
  <name>_predictions_probabilities.tsv            Tab-separated file with  class probability columns indexed by sample ID, and also with 'truth' and 'prediction' columns for each sample.
  <name>_classification_report.tsv  Tab-separated file summarizing precision, recall, f1-score for each class.
  <name>_confusion_matrix.tsv       Tab-separated file containing the confusion matrix (actual vs. predicted).
  <name>_feature_importances.tsv    Tab-separated file of feature importance scores averaged across folds (if available).
  <name>_shap_values.npy            NumPy array file of aggregated SHAP values across folds (if computed).
  <name>_confusion_matrix.png       Heatmap of the confusion matrix saved as PNG.
  <name>_shap_summary.png           SHAP summary beeswarm plot saved as PNG (if SHAP values computed).

Instructions of Use:
  This script performs grouped cross-validation or single-run evaluation of a scikit-learn/imblearn
  pipeline model. It loads the model and features table, executes training and/or prediction,
  computes metrics and explanations, and writes results to the specified output directory.

Usage Example:
  python 05_evaluate_model.py \
    --model path/to/model.joblib \
    --features path/to/features.tsv \
    --label target_column \
    --group_column group_column \
    --n_splits 5 \
    --output_dir results/ \
    --name experiment1 \
    --log_level INFO
    --fasta
"""

# ────────────────────────────── standard library ─────────────────────────────
from __future__ import annotations

import argparse                      # CLI parsing
import logging                       # console logging
import sys                           # graceful exits
from collections import defaultdict  # group counting helper
from pathlib import Path             # path handling
from typing import Optional, List

# ──────────────────────────────── 3rd‑party ──────────────────────────────────
import joblib                        # load joblib artefact
import matplotlib.pyplot as plt      # plots
import numpy as np                   # numerical ops
import pandas as pd                  # TSV I/O
import shap                          # SHAP explanations
from sklearn.base import clone       # deep‑copy estimator
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

# ─────────────────────────────── global consts ───────────────────────────────
RSEED = 50
plt.rcParams.update({"figure.autolayout": True})

# ╭──────────────────────────────────────────────────────────────────────────╮
# │                               utilities                                 │
# ╰──────────────────────────────────────────────────────────────────────────╯


def infer_problem_type(y: np.ndarray) -> str:
    """Return `'muticlass'` (>2 classes) or `'binary'`."""
    return "multiclass" if len(np.unique(y)) > 2 else "binary"


def min_groups_per_class(y: np.ndarray, groups: np.ndarray) -> int:
    """Smallest number of distinct groups containing any class."""
    c2g = defaultdict(set)
    for label, grp in zip(y, groups):
        c2g[label].add(grp)
    return min(len(g) for g in c2g.values())


def get_cv_iterator(y: np.ndarray, groups: np.ndarray, n_splits):
    """Return list of train/test indices for grouped CV; error if impossible."""
    min_g = min_groups_per_class(y, groups)
    if n_splits > min_g:
        raise ValueError(
            f"n_splits={n_splits} > available groups per rarest class ({min_g})")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RSEED)
    return list(sgkf.split(np.zeros_like(y), y, groups))

def predict_with_pipeline(pipeline, X):
    """
    Return (pred_labels, pred_probabilities or None).

    • Aligns column order for XGBoost.
    • Works with any scikit-learn classifier that implements predict_proba().
    """
    model = pipeline.named_steps.get("model", pipeline)

    # ---------- XGBoost branch -------------------------------------------
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        expected = booster.feature_names

        missing = set(expected) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features for prediction: {missing}")

        extra = set(X.columns) - set(expected)
        if extra:
            X = X.drop(columns=list(extra))

        X = X[expected]

        preds  = pipeline.predict(X)
        probas = pipeline.predict_proba(X)          # keep full matrix
        return preds, probas

    # ---------- Generic sklearn branch -----------------------------------
    preds = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X)          # full matrix
    else:
        probas = None

    return preds, probas
# ╭──────────────────────────────────────────────────────────────────────────╮
# │                          core evaluation loop                           │
# ╰──────────────────────────────────────────────────────────────────────────╯

def run_evaluation(
    model_path: Path,
    features_tsv: Path,
    label_col: str,
    group_col: str,
    n_splits: Optional[int],
    no_cv: bool,
    output_dir: Path,
    name: str,
    fasta: bool,
    predict_only: bool = False
) -> None:
    """Grouped‑CV evaluation: predictions, metrics, feature importances, SHAP."""

    # 0️⃣  Output directory
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣  Load artefacts & data ------------------------------------------------
    logging.info("Loading model ➜ %s", model_path)
    pipeline = joblib.load(model_path)  # imblearn.Pipeline or estimator

    logging.info("Reading feature matrix ➜ %s", features_tsv)
    df = pd.read_csv(features_tsv, sep="\t", index_col=0)

    if predict_only:
        logging.info("Prediction-only mode enabled. Skipping evaluation.")
        X = df.copy()
        preds = pipeline.predict(X)
        pred_df = pd.DataFrame({
            "IsolateID": X.index,
            "Prediction": preds
        })
        output_file = output_dir / f"{name}_predictions.tsv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_file, sep="\t", index=False)
        logging.info("Predictions written to ➜ %s", output_file)
        return


    if label_col not in df.columns or group_col not in df.columns:
        raise KeyError("label or group column missing in TSV header")

    # Encode string labels to integers
    le = LabelEncoder()
    y_raw = df[label_col].values
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    X = df.drop(columns=[label_col, group_col])
    groups = df[group_col].values

    problem_type = infer_problem_type(y)

    # containers for results
    oof_true, oof_pred, oof_proba = [], [], []
    feature_imps, shap_vals_all = [], []

    if no_cv:
        logging.info("Hold‑out mode (no CV)")
        preds, probas = predict_with_pipeline(pipeline, X)
        oof_true.append(y)
        oof_pred.append(preds)
        oof_proba.append(
            pd.DataFrame(probas, index=X.index, columns=class_names)
        )
        #model_step = pipeline.named_steps.get('model', pipeline)
        if hasattr(pipeline, 'named_steps'):
            # sklearn Pipeline – get the 'model' step or fall back
            model_step = pipeline.named_steps.get('model', pipeline)
        else:
            # Raw classifier instance
            model_step = pipeline

        if hasattr(model_step, 'feature_importances_'):
            feature_imps.append(
                pd.Series(model_step.feature_importances_, index=X.columns)
            )
        shap_vals_all.append(shap.TreeExplainer(model_step).shap_values(X))
    else:
        if n_splits is None:
            raise ValueError("n_splits required when CV enabled")
    # 2️⃣  Build CV iterator ----------------------------------------------------
        cv_iter = get_cv_iterator(y, groups, n_splits)

        # 3️⃣  Fold loop ------------------------------------------------------------
        for fold, (tr, te) in enumerate(cv_iter, 1):
            logging.info("Fold %d/%d", fold, n_splits)
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y[tr], y[te]

            # clone keeps original hyper‑params but clears fitted state
            clf = clone(pipeline)

            # detect if pipeline has named steps (oversampler/model) or raw estimator
            named = hasattr(clf, 'named_steps')
            uses_oversampler = named and 'oversampler' in clf.named_steps
            if named:
                model_step = clf.named_steps.get('model', clf)
            else:
                model_step = clf
            is_xgb = model_step.__class__.__name__.startswith('XGB')

            # balanced sample‑weights for XGBoost when *no* oversampling step exists
            fit_kwargs = {}
            if is_xgb and not uses_oversampler:
                key = 'model__sample_weight' if named else 'sample_weight'
                fit_kwargs[key] = compute_sample_weight('balanced', y_tr)

            clf.fit(X_tr, y_tr, **fit_kwargs)

            preds, probas = predict_with_pipeline(clf, X_te)

            oof_true.append(y_te)
            oof_pred.append(preds)
            oof_proba.append(pd.DataFrame(probas, index=X_te.index, columns=class_names))


            # feature importances
            model_step = clf.named_steps.get("model", clf)
            if hasattr(model_step, "feature_importances_"):
                feature_imps.append(pd.Series(model_step.feature_importances_, index=X.columns))

            # SHAP values
            shap_vals_all.append(shap.TreeExplainer(model_step).shap_values(X_te))

    # 4️⃣  Aggregate OOF results -----------------------------------------------
    y_true = np.concatenate(oof_true)
    y_pred = np.concatenate(oof_pred)

    # Map integers back to original labels
    if np.issubdtype(y_pred.dtype, np.integer):
        y_pred_str = le.inverse_transform(y_pred)
        y_true_str = le.inverse_transform(y_true)
    else:
        # assume they’re already the true‐label strings
        y_pred_str = y_pred
        y_true_str = le.inverse_transform(y_true)

    proba_df = pd.concat(oof_proba)

    acc = accuracy_score(y_true_str, y_pred_str)
    bacc = balanced_accuracy_score(y_true_str, y_pred_str)
    logging.info("OOF Accuracy %.4f | Balanced Accuracy %.4f", acc, bacc)

    report_df = pd.DataFrame(classification_report(y_true_str, y_pred_str, output_dict=True)).T.reset_index()
    cm_arr = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    cm_df = pd.DataFrame(cm_arr, index=class_names, columns=class_names)

    if feature_imps:
        fi_df = pd.concat(feature_imps, axis=1)
        fi_df["average"] = fi_df.mean(axis=1)
        fi_df = fi_df.sort_values("average", ascending=False)
    else:
        fi_df = pd.DataFrame()

    #  ➡️  Optional: write FASTA of features by importance

    if fasta and not fi_df.empty:
        fasta_path = output_dir / f"{name}_features.fasta"
        with open(fasta_path, "w") as fh:
            for rank, feature in enumerate(fi_df.index, start=1):
                fh.write(f">Feature_{rank}\n")
                fh.write(f"{feature}\n")
        logging.info("Written FASTA of features ➜ %s", fasta_path)

 #   combine SHAP arrays across folds
    shap_stack = None
    if shap_vals_all:
        shap_stack = (
            [np.concatenate([fold[c] for fold in shap_vals_all], axis=0) for c in range(len(class_names))]
            if isinstance(shap_vals_all[0], list)
            else np.concatenate(shap_vals_all, axis=0)
    )


    # 5️⃣  Write artefacts ------------------------------------------------------
        # Prepare files dict for downstream references
    files = {
        "predictions_probabilities": output_dir / f"{name}_predictions_probabilities.tsv",
        "class_report": output_dir / f"{name}_classification_report.tsv",
        "conf_matrix": output_dir / f"{name}_confusion_matrix.tsv",
        "feat_importances": output_dir / f"{name}_feature_importances.tsv",
        "shap_values": output_dir / f"{name}_shap_values.npy",
    }

    # Combine predictions and probabilities into a single TSV
    combined_df = proba_df.copy()
    combined_df["truth"] = y_true_str
    combined_df["prediction"] = y_pred_str
    combined_df = combined_df.reset_index().rename(columns={"index": "IsolateID"})
    files["predictions_probabilities"].parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(files["predictions_probabilities"], sep="\t", index=False)

    report_df.to_csv(files["class_report"], sep="\t", index=False)
    cm_df.to_csv(files["conf_matrix"], sep="\t")
    if not fi_df.empty:
        fi_df.to_csv(files["feat_importances"], sep="\t")
    if shap_stack is not None:
        np.save(files["shap_values"], shap_stack)
        # Save per-class SHAP values as TSV files
        if isinstance(shap_stack, list):
            # Multiclass: one array per class
            for idx, cls in enumerate(class_names):
                df_shap = pd.DataFrame(
                    shap_stack[idx], index=proba_df.index, columns=X.columns)
                df_shap.to_csv(output_dir / f"{name}_{cls}_shap_values.tsv", sep="\t")
        else:
            # Binary (single 2D array): save under both class names
            for cls in class_names:
                df_shap = pd.DataFrame(shap_stack, index=proba_df.index, columns=X.columns)
                df_shap.to_csv(output_dir / f"{name}_{cls}_shap_values.tsv", sep="\t")


    # 6️⃣  Diagnostic plots -----------------------------------------------------
    # confusion matrix heat‑map
    ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=class_names).plot(
        cmap=plt.cm.Blues,
        values_format=".2g",
    )
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    # SHAP summary beeswarm
    if shap_stack is not None:
        for idx, cls in enumerate(class_names):
            shap.summary_plot(
                shap_stack[idx],  # SHAP values for this class
                X.loc[proba_df.index],
                feature_names=X.columns,
                show=False,
            )
            plt.savefig(output_dir / f"{name}_shap_summary_{cls}.png", dpi=300)
            plt.close()

    logging.info("All outputs written to %s", output_dir)

# ╭──────────────────────────────────────────────────────────────────────────╮
# │                                 CLI                                    │
# ╰──────────────────────────────────────────────────────────────────────────╯

def parse_args():
    p = argparse.ArgumentParser("Grouped‑CV evaluation (TSV outputs only)")
    p.add_argument("--model", type=Path, required=True, help="trained *.joblib artefact")
    p.add_argument("--features", type=Path, required=True, help="TSV with features + label + group")
    p.add_argument("--label", required=False, help="name of the label column")
    p.add_argument("--group_column", required=False, help="name of the group column")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_splits", type=int, help="CV folds (default 5)")
    group.add_argument('--no_cv', action='store_true', help='Skip CV and predict full dataset once')
    p.add_argument("--output_dir", type=Path, required=True, help="directory for outputs")
    p.add_argument("--name", required=True, help="prefix for output files")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--fasta", action="store_true", help = "If set, write a FASTA file of features sorted by importance")
    p.add_argument("--predict_only", action="store_true",help="Only output predictions without evaluating performance.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        run_evaluation(
            model_path=args.model,
            features_tsv=args.features,
            label_col=args.label,
            group_col=args.group_column,
            n_splits=args.n_splits,
            no_cv=args.no_cv,
            output_dir=args.output_dir,
            name=args.name,
            fasta=args.fasta,
            predict_only=args.predict_only
        )
    except Exception:
        logging.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
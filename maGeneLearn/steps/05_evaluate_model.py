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
  --skip-shap      If set, skip SHAP value computation (faster for large datasets).
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
from collections import defaultdict, Counter  # group counting helper
from pathlib import Path             # path handling
from typing import Optional, List
from datetime import datetime

# ──────────────────────────────── 3rd‑party ──────────────────────────────────
import joblib                        # load joblib artefact
import matplotlib.pyplot as plt      # plots
import numpy as np                   # numerical ops
import pandas as pd                  # TSV I/O
import shap                          # SHAP explanations
from sklearn.base import clone       # deep‑copy estimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    average_precision_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

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

    if hasattr(model, "feature_names_in_"):
        # Align dataframe to training column order before both predict()
        # and predict_proba().
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    preds = pipeline.predict(X)

    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(X)  # full matrix
    else:
        probas = None

    return preds, probas

def build_probability_df(probas, index, model_step, le, class_names):
    """
    Build a probability DataFrame with columns aligned to the global class names.

    Handles cases where the fitted estimator only knows a subset of classes
    because a sampler such as ENN/SMOTEENN removed one class during training.
    """
    if probas is None:
        return None

    if hasattr(model_step, "classes_"):
        model_classes = np.asarray(model_step.classes_)
    else:
        model_classes = np.arange(probas.shape[1])

    # Convert encoded model classes back to original labels when possible.
    if (
        le is not None
        and hasattr(le, "classes_")
        and np.issubdtype(model_classes.dtype, np.integer)
    ):
        model_classes = le.inverse_transform(model_classes.astype(int))

    proba_cols = [str(c) for c in model_classes]

    if len(proba_cols) != probas.shape[1]:
        raise ValueError(
            f"Number of fitted model classes ({len(proba_cols)}) does not match "
            f"predict_proba output shape ({probas.shape[1]}). "
            f"Model classes: {proba_cols}"
        )

    proba_df = pd.DataFrame(
        probas,
        index=index,
        columns=proba_cols,
    )

    all_classes = [str(c) for c in class_names]

    missing_classes = [c for c in all_classes if c not in proba_df.columns]
    if missing_classes:
        logging.warning(
            "Model predict_proba() returned probabilities for only %d/%d classes. "
            "Missing classes will be filled with probability 0.0: %s",
            len(proba_df.columns),
            len(all_classes),
            missing_classes,
        )

    proba_df = proba_df.reindex(columns=all_classes, fill_value=0.0)

    return proba_df
# ╭──────────────────────────────────────────────────────────────────────────╮
# │                          core evaluation loop                           │
# ╰──────────────────────────────────────────────────────────────────────────╯

def setup_logging(output_dir: Path, name: str, log_level: str) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"{name}_evaluate_{timestamp}.log"

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return log_file


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
    scoring: str,
    predict_only: bool = False,
    skip_svm_importance: bool = False,
    skip_shap: bool = False
) -> None:
    """Grouped‑CV evaluation: predictions, metrics, feature importances, SHAP."""

    # 0️⃣  Output directory
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣  Load artefacts & data ------------------------------------------------
    logging.info("Loading model ➜ %s", model_path)
    pipeline = joblib.load(model_path)  # imblearn.Pipeline or estimator

    # ---------------------------------------------------------------
    # Handle case where model was saved as {"model": pipeline, "label_encoder": le}
    # ---------------------------------------------------------------
    if isinstance(pipeline, dict) and "label_encoder" in pipeline:
        le_model = pipeline["label_encoder"]
        pipeline = pipeline["model"]

        if le_model is not None:
            logging.info(
                "Loaded LabelEncoder with classes: %s",
                [str(c) for c in le_model.classes_]
            )
            logging.info(
                "Preserving fitted model classes_ for prediction/probability alignment."
            )

    # ---------------------------------------------------------------
    # Handle legacy models (no encoder, numeric classes)
    # ---------------------------------------------------------------
    else:
        model_step = (
            pipeline.named_steps["model"]
            if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps
            else pipeline
        )
        if hasattr(model_step, "classes_") and np.issubdtype(type(model_step.classes_[0]), np.number):
            logging.warning("Old model detected – converting numeric classes to strings.")
            model_step.__dict__["classes_"] = np.array([str(c) for c in model_step.classes_])

    # ---------------------------------------------------------------
    # Build consistent LabelEncoder from the model’s classes
    # ---------------------------------------------------------------
    # Only rebuild the encoder if we *didn't* load one from the artifact
    if "le_model" in locals() and le_model is not None:
        le = le_model
        class_names = list(le.classes_)
        logging.info("Using loaded LabelEncoder (classes: %s)", class_names)
    else:
        model_step = (
            pipeline.named_steps["model"]
            if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps
            else pipeline
        )
        le = LabelEncoder()
        if hasattr(model_step, "classes_"):
            le.classes_ = np.array(model_step.classes_)
            class_names = list(model_step.classes_)
            logging.info("LabelEncoder rebuilt from model classes: %s", class_names)
        else:
            le = None
            class_names = []

    #Load feature matrix
    logging.info("Reading feature matrix ➜ %s", features_tsv)
    df = pd.read_csv(features_tsv, sep="\t", index_col=0)

    if predict_only:
        logging.info("Prediction-only mode enabled. Skipping evaluation.")
        X = df.copy()

        preds, probas = predict_with_pipeline(pipeline, X)
        preds = np.asarray(preds)

        # Convert encoded predictions back to original labels when needed.
        # This fixes XGBoost models saved with a LabelEncoder.
        if le is not None and hasattr(le, "classes_") and np.issubdtype(preds.dtype, np.integer):
            logging.info("Converting encoded predictions back to original labels.")
            preds = le.inverse_transform(preds)

        pred_df = pd.DataFrame({
            "IsolateID": X.index,
            "Prediction": preds
        })

        # Add class probabilities if the model supports predict_proba().
        # Add class probabilities if the model supports predict_proba().
        if probas is not None:
            logging.info("Adding class probability columns to prediction output.")

            model_step = (
                pipeline.named_steps["model"]
                if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps
                else pipeline
            )

            proba_df = build_probability_df(
                probas=probas,
                index=X.index,
                model_step=model_step,
                le=le,
                class_names=class_names,
            )

            pred_df = pd.concat(
                [pred_df.set_index("IsolateID"), proba_df],
                axis=1
            ).reset_index()
        else:
            logging.warning(
                "Model does not expose predict_proba(); writing predictions only."
            )

        output_file = output_dir / f"{name}_predictions.tsv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(output_file, sep="\t", index=False)
        logging.info("Predictions written to ➜ %s", output_file)
        return

    # ------------------------------
    # Column requirements:
    # - label is required for any evaluation (predict_only handled above)
    # - group is required ONLY for CV splitting
    # - in --no_cv, group may be absent; if present it will still be dropped from X
    # ------------------------------

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' missing in TSV header")

    if (not no_cv) and (group_col not in df.columns):
        raise KeyError(f"Group column '{group_col}' missing in TSV header (required for grouped CV)")


    # 3️⃣  Encode labels --------------------------------------------------------
    y_raw = df[label_col].values
    if le is not None and hasattr(le, "classes_"):
        # --- Diagnostic logging: check what the encoder sees ---
        logging.info("Encoder classes: %s", list(le.classes_))
        logging.info("Unique labels in test data: %s", np.unique(y_raw))

        unmatched = [lbl for lbl in np.unique(y_raw) if str(lbl) not in [str(c) for c in le.classes_]]
        if unmatched:
            logging.warning("Labels in test data not found in model classes: %s", unmatched)

        # --- Make sure everything is compared as strings ---
        le.classes_ = np.array([str(c) for c in le.classes_])
        class_to_idx = {cls: i for i, cls in enumerate(le.classes_)}

        y = np.array([class_to_idx.get(str(label), -1) for label in y_raw])
    else:
        le = LabelEncoder().fit(y_raw)
        y = le.transform(y_raw)
        class_names = list(le.classes_)
        logging.info("LabelEncoder fitted directly on input labels: %s", class_names)

#    X = df.drop(columns=[label_col, group_col])
#    groups = df[group_col].values
    # Always drop label (and group if present) so they never leak into model features.
    drop_cols = [label_col]
    if group_col in df.columns:
        drop_cols.append(group_col)
    X = df.drop(columns=drop_cols)

    # groups only needed for CV splitting
    groups = df[group_col].values if (not no_cv) else None


    problem_type = infer_problem_type(y)

    # containers for results
    oof_true, oof_pred, oof_proba = [], [], []
    feature_imps, shap_vals_all = [], []

    # SHAP is optional. If it fails in any fold, evaluation should continue
    # and all non-SHAP outputs should still be written.
    shap_failed = False

    if no_cv:
        logging.info("Hold‑out mode (no CV)")
        preds, probas = predict_with_pipeline(pipeline, X)
        oof_true.append(y)
        oof_pred.append(preds)
        model_step = (
            pipeline.named_steps["model"]
            if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps
            else pipeline
        )

        proba_df = build_probability_df(
            probas=probas,
            index=X.index,
            model_step=model_step,
            le=le,
            class_names=class_names,
        )

        oof_proba.append(proba_df)
        #model_step = pipeline.named_steps.get('model', pipeline)
        if hasattr(pipeline, 'named_steps'):
            # sklearn Pipeline – get the 'model' step or fall back
            model_step = pipeline.named_steps.get('model', pipeline)
        else:
            # Raw classifier instance
            model_step = pipeline

        # Skip feature importance and SHAP in test-set evaluation
        logging.info("Skipping feature importance and SHAP in test evaluation mode.")
    else:
        if n_splits is None:
            raise ValueError("n_splits required when CV enabled")
    # 2️⃣  Build CV iterator ----------------------------------------------------
 #       cv_iter = get_cv_iterator(y, groups, n_splits)
        if groups is None:
            raise RuntimeError("Internal error: groups is None while CV is enabled")
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
            uses_sampler = named and (
                'sampler' in clf.named_steps or
                'oversampler' in clf.named_steps)

            if named:
                model_step = clf.named_steps.get('model', clf)
            else:
                model_step = clf
            is_xgb = model_step.__class__.__name__.startswith('XGB')

            # balanced sample‑weights for XGBoost when *no* oversampling step exists
            fit_kwargs = {}
            if is_xgb and not uses_sampler:
                key = 'model__sample_weight' if named else 'sample_weight'
                fit_kwargs[key] = compute_sample_weight('balanced', y_tr)

            try:
                clf.fit(X_tr, y_tr, **fit_kwargs)

            except Exception as e:
                before_counts = Counter(y_tr)

                sampler_name = "none"
                after_counts = None

                if hasattr(clf, "named_steps"):
                    if "sampler" in clf.named_steps:
                        sampler_name = clf.named_steps["sampler"].__class__.__name__
                    elif "oversampler" in clf.named_steps:
                        sampler_name = clf.named_steps["oversampler"].__class__.__name__

                # Try to diagnose what the sampler did.
                # This is only diagnostic; if it also fails, we still raise the original error.
                if hasattr(clf, "named_steps") and "sampler" in clf.named_steps:
                    try:
                        X_res_dbg, y_res_dbg = clf.named_steps["sampler"].fit_resample(X_tr, y_tr)
                        after_counts = Counter(y_res_dbg)
                    except Exception as sampler_e:
                        after_counts = f"Sampler diagnostic failed: {sampler_e}"

                logging.error(
                    "Model fitting failed in fold %d/%d. "
                    "Model=%s | Sampler=%s | Train shape=%s | "
                    "Class counts before sampling=%s | Class counts after sampling=%s",
                    fold,
                    n_splits,
                    model_step.__class__.__name__,
                    sampler_name,
                    X_tr.shape,
                    dict(before_counts),
                    dict(after_counts) if isinstance(after_counts, Counter) else after_counts,
                )

                if (
                        "num_class" in str(e)
                        or "The target 'y' needs to have more than 1 class" in str(e)
                        or "Found array with 0 sample" in str(e)
                ):
                    raise RuntimeError(
                        f"Model fitting failed in fold {fold}/{n_splits}. "
                        f"This is likely caused by the sampler step ({sampler_name}) producing "
                        "an invalid training set for this fold, for example zero samples or too few "
                        "classes after resampling/cleaning. This can happen with aggressive samplers "
                        "such as SMOTEENN/ENN in grouped CV, especially when feature selection leaves "
                        "few features or some classes are sparse. See log above for class counts before "
                        "and after sampling."
                    ) from e

                raise

            preds, probas = predict_with_pipeline(clf, X_te)

            model_step = (
                clf.named_steps["model"]
                if hasattr(clf, "named_steps") and "model" in clf.named_steps
                else clf
            )

            proba_df = build_probability_df(
                probas=probas,
                index=X_te.index,
                model_step=model_step,
                le=le,
                class_names=class_names,
            )

            oof_true.append(y_te)
            oof_pred.append(preds)
            oof_proba.append(proba_df)


            # feature importances
            model_step = clf.named_steps.get("model", clf)
            if hasattr(model_step, "feature_importances_"):
                feature_imps.append(pd.Series(model_step.feature_importances_, index=X.columns))

            elif isinstance(model_step, LogisticRegression):
                logging.info("Extracting coefficients as feature importance for Logistic Regression.")
                coefs = np.mean(np.abs(model_step.coef_), axis=0)
                fi = pd.Series(coefs, index=X.columns)
                feature_imps.append(fi)

            elif model_step.__class__.__name__ == "SVC":
                if skip_svm_importance:
                    logging.info("Skipping permutation importance for SVM because --skip-svm-importance was set.")
                else:
                    logging.info("Computing permutation importance for SVM (subset of features).")
                    if hasattr(model_step, "feature_names_in_"):
                        X_te_aligned = X_te.reindex(columns=model_step.feature_names_in_, fill_value=0)
                    else:
                        X_te_aligned = X_te
                    # Reduce to top-N features by univariate variance (or just first N cols)
                    N_TOP = min(500, X_te.shape[1])  # cap at 500 features
                    top_features = X_te.iloc[:, :N_TOP]  # simple subset; could replace with chi² selection
                    result = permutation_importance(
                        clf, top_features, y_te,
                        n_repeats=10,
                        random_state=RSEED,
                        n_jobs=-1,
                        scoring=scoring
                    )
                    fi = pd.Series(result.importances_mean, index=top_features.columns)
                    feature_imps.append(fi)
            else:
                logging.info("Skipping feature importance: model type not supported (%s)", type(model_step))

            if hasattr(model_step, "feature_names_in_"):
                X_te = X_te.reindex(columns=model_step.feature_names_in_, fill_value=0)

            # SHAP values only for tree-based models
            if (
                    not skip_shap
                    and not shap_failed
                    and (
                    model_step.__class__.__name__.startswith("XGB")
                    or hasattr(model_step, "feature_importances_")
            )
            ):
                try:
                    # --- Diagnostic: check model vs. data alignment ---
                    logging.info(
                        "DEBUG SHAP | model: %s | X_te shape: %s | n_features_in_: %s",
                        type(model_step).__name__,
                        X_te.shape,
                        getattr(model_step, "n_features_in_", "NA")
                    )

                    if hasattr(model_step, "feature_names_in_"):
                        model_feats = list(model_step.feature_names_in_)
                        diff1 = set(model_feats) - set(X_te.columns)
                        diff2 = set(X_te.columns) - set(model_feats)
                        logging.info(
                            "DEBUG SHAP | missing_in_X_te=%d | extra_in_X_te=%d",
                            len(diff1),
                            len(diff2)
                        )
                        if diff1:
                            logging.warning("Features missing in X_te: %s", list(diff1)[:10])
                        if diff2:
                            logging.warning("Extra features in X_te: %s", list(diff2)[:10])
                    # --- End of diagnostic block ---

                    #Shap computation
                    explainer = shap.TreeExplainer(model_step)
                    shap_vals = explainer.shap_values(X_te)

                    # Align SHAP outputs to the original class labels.
                    # This is needed because fitted models may store classes as
                    # encoded integers, while class_names contains original labels.
                    fold_classes_2 = np.asarray(model_step.classes_)

                    if (
                        le is not None
                        and hasattr(le, "classes_")
                        and np.issubdtype(fold_classes_2.dtype, np.integer)
                    ):
                        fold_classes_2 = le.inverse_transform(fold_classes_2.astype(int))

                    fold_classes_2 = [str(c) for c in fold_classes_2]
                    target_classes = [str(c) for c in class_names]

                    shap_dict = {}

                    if isinstance(shap_vals, list):
                        # Common multiclass format: one SHAP matrix per fitted class
                        for i, cls in enumerate(fold_classes_2):
                            if i < len(shap_vals):
                                shap_dict[cls] = shap_vals[i]

                    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                        # Some SHAP versions return a 3D array:
                        # either (n_samples, n_features, n_classes)
                        # or occasionally (n_classes, n_samples, n_features).
                        if shap_vals.shape[2] == len(fold_classes_2):
                            for i, cls in enumerate(fold_classes_2):
                                shap_dict[cls] = shap_vals[:, :, i]

                        elif shap_vals.shape[0] == len(fold_classes_2):
                            for i, cls in enumerate(fold_classes_2):
                                shap_dict[cls] = shap_vals[i, :, :]

                        else:
                            raise ValueError(
                                "Could not confidently align 3D SHAP values. "
                                f"SHAP shape={shap_vals.shape} | "
                                f"fitted classes={fold_classes_2}"
                            )

                    else:
                        # Binary/single-output case. The returned matrix corresponds
                        # to one model output, so attach it to the last fitted class.
                        # Missing classes below will receive zero SHAP values.
                        target_cls = fold_classes_2[-1] if fold_classes_2 else target_classes[-1]
                        shap_dict[target_cls] = shap_vals

                    aligned = []

                    for cls in target_classes:
                        if cls in shap_dict:
                            aligned.append(shap_dict[cls])
                        else:
                            # Create zero SHAP values for classes absent from this fold/model.
                            aligned.append(np.zeros((X_te.shape[0], X_te.shape[1])))

                    shap_vals_all.append(aligned)

                except Exception:
                    logging.exception(
                        "SHAP failed in fold %d/%d. "
                        "SHAP outputs will be skipped, but model evaluation "
                        "will continue.",
                        fold,
                        n_splits,
                    )

                    # Do not retain partial SHAP results from earlier folds.
                    shap_failed = True
                    shap_vals_all.clear()


            elif isinstance(model_step, LogisticRegression):
                logging.info("Skipping SHAP: Logistic Regression not supported (use coefficients instead).")

            elif shap_failed:
                logging.info(
                    "Skipping SHAP in fold %d because SHAP failed "
                    "in an earlier fold.",
                    fold,
                )

            elif skip_shap:
                logging.info("Skipping SHAP because --skip-shap was set.")

            else:
                logging.info("Skipping SHAP: model type not supported (%s)", type(model_step))


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

    report_dict = classification_report(y_true_str, y_pred_str, output_dict=True, zero_division=0)
    report_df= pd.DataFrame(report_dict).T.reset_index()

    # Adjust macro avg to include only classes with support > 0
    # ────────────────────────────────────────────────
    # Identify only those rows that correspond to actual classes
    mask_classes = report_df["index"].isin(np.unique(y_true_str))
    nonzero_classes = report_df[mask_classes & (report_df["support"] > 0)]

    if not nonzero_classes.empty:
        macro_prec = nonzero_classes["precision"].mean()
        macro_rec = nonzero_classes["recall"].mean()
        macro_f1 = nonzero_classes["f1-score"].mean()

        report_df.loc[report_df["index"] == "macro avg", ["precision", "recall", "f1-score"]] = [macro_prec, macro_rec, macro_f1]

    # Optional cleanup: remove zero-support classes for cleaner reporting
    report_df = report_df[~((report_df["support"] == 0) & (report_df["index"].isin(le.classes_)))]

    cm_arr = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    cm_df = pd.DataFrame(cm_arr, index=class_names, columns=class_names)

    # ───────────────────────────── MCC and AUPRC ─────────────────────────────
    logging.info("Computing additional metrics: MCC and AUPRC (per-class, macro, micro)")

    mcc = matthews_corrcoef(y_true_str, y_pred_str)
    # ------------------------------------------------------------------
    # Build one-vs-rest binary truth matrix explicitly.
    #
    # Do NOT use label_binarize() here because for binary classification
    # sklearn returns only one column rather than one column per class.
    #
    # This produces:
    #     n_samples x n_classes
    # for both binary and multiclass problems.
    # ------------------------------------------------------------------
    class_names_str = [str(cls) for cls in class_names]
    y_true_str_arr = np.asarray(y_true_str).astype(str)

    y_true_bin = np.column_stack([
        (y_true_str_arr == cls).astype(int)
        for cls in class_names_str
    ])

    # Probability columns were already aligned to the global class order
    # by build_probability_df().
    y_score = proba_df[class_names_str].to_numpy()

    # --- Per-class AUPRC ---
    auprc_per_class = {
        cls: average_precision_score(
            y_true_bin[:, i],
            y_score[:, i]
        )
        for i, cls in enumerate(class_names_str)
    }

    # --- Macro and micro AUPRC ---
    auprc_macro = np.mean(list(auprc_per_class.values()))
    auprc_micro = average_precision_score(
        y_true_bin.ravel(),
        y_score.ravel()
    )

    # --- Assemble results ---
    auprc_mcc = {
        "MCC": round(mcc, 4),
        "Macro_AUPRC": round(auprc_macro, 4),
        "Micro_AUPRC": round(auprc_micro, 4),
        **{
            f"AUPRC_{cls}": round(v, 4)
            for cls, v in auprc_per_class.items()
        },
    }

    auprc_mcc_df = pd.DataFrame([auprc_mcc])
    auprc_mcc_path = output_dir / f"{name}_mcc_auprc.tsv"
    auprc_mcc_df.to_csv(auprc_mcc_path, sep="\t", index=False)
    logging.info("Extra metrics written to ➜ %s", auprc_mcc_path)


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


    # 5️⃣  Write core results ------------------------------------------------------
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


    # 6️⃣  Diagnostic plots -----------------------------------------------------
    # confusion matrix heat‑map
    ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=class_names).plot(
        cmap=plt.cm.Blues,
        values_format=".2g",
    )
    plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=300)
    plt.close()

    #Optional SHAP outputs
    #   combine SHAP arrays across folds
    shap_stack = None

    if shap_failed:
        logging.warning(
            "SHAP failed during cross-validation. "
            "All SHAP outputs will be skipped."
        )

    elif shap_vals_all:

        try:
            logging.info("Combining SHAP values across CV folds.")
            shap_stack = (
                [np.concatenate([fold[c] for fold in shap_vals_all], axis=0) for c in range(len(class_names))]
                if isinstance(shap_vals_all[0], list)
                else np.concatenate(shap_vals_all, axis=0)
            )

            # Save raw SHAP values
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

            # SHAP summary beeswarm
            if isinstance(shap_stack, list):
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

        except Exception:
            logging.exception(
                "SHAP post-processing failed. "
                "Core evaluation outputs were already written and "
                "will be preserved."
            )

    else:
        logging.info("No SHAP values available; skipping SHAP outputs.")

    logging.info("All core evaluation outputs written to %s", output_dir)


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
    p.add_argument("--scoring",  type=str,help="Scoring parameter for best model")
    p.add_argument("--skip-svm-importance", action="store_true", help="Skip permutation importance calculation for SVM models.")
    p.add_argument("--skip-shap", action="store_true", help="Skip SHAP value computation.")
    return p.parse_args()


def main():
    args = parse_args()

    log_file = setup_logging(
        args.output_dir,
        args.name,
        args.log_level
    )

    logging.info("Writing evaluation log to %s", log_file)

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
            predict_only=args.predict_only,
            scoring=args.scoring,
            skip_svm_importance=args.skip_svm_importance,
            skip_shap=args.skip_shap,
        )
    except Exception:
        logging.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

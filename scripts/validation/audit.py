#!/usr/bin/env python3
"""Independently audit saved-model indices, including fitted-class probability values."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

from common import checksum
from matrix import check_artifact_probabilities, check_evaluation, frame, one, require


def check_tuning(directory):
    """Require real optimization output and independently verify fold aggregates."""
    import numpy as np
    trials = frame(one(directory, "*.tsv"))
    require(trials.index.is_unique and 1 <= len(trials) <= 3, "Invalid optimization trial history")
    columns = ["fold0_score", "fold1_score", "fold2_score"]
    scores = trials[columns].to_numpy(dtype=float)
    valid = np.isfinite(scores).all(axis=1)
    require(valid.any(), "Optimization has no finite three-fold trial")
    np.testing.assert_allclose(trials.loc[valid, "score_mean"], scores[valid].mean(axis=1), atol=1e-10)
    np.testing.assert_allclose(trials.loc[valid, "score_std"], scores[valid].std(axis=1), atol=1e-10)


def check_configuration(fitted, model, sampling):
    """Require the requested CLI branches in the saved pipeline itself."""
    expected_models = {"RFC": "RandomForestClassifier", "XGBC": "XGBClassifier", "SVM": "SVC", "LR": "LogisticRegression"}
    expected_samplers = {"none": None, "random": "RandomOverSampler", "smote": "SMOTE",
                         "random_under": "RandomUnderSampler", "enn": "EditedNearestNeighbours", "smoteenn": "SMOTEENN"}
    require(hasattr(fitted, "named_steps"), "New model artifact omitted fitted pipeline")
    require(type(fitted.named_steps["model"]).__name__ == expected_models[model], "Saved classifier differs from requested model")
    sampler = fitted.named_steps.get("sampler")
    actual = type(sampler).__name__ if sampler is not None else None
    require(actual == expected_samplers[sampling], "Saved sampler differs from requested sampling setting")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--model-index", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]:
        os.environ[name] = "1"
    import numpy as np
    results, seen = [], set()
    report = {"python": sys.version, "script_sha256": checksum(Path(__file__)),
              "indices": {str(p): checksum(p) for p in args.model_index}, "results": results}
    for index in args.model_index:
        with index.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            if row["model_file"] in seen:
                continue
            seen.add(row["model_file"])
            result = {"artifact": row["model_file"], "problem": row["problem"], "model": row["model"],
                      "selection": row["selection"], "sampling": row["sampling"]}
            try:
                require(checksum(row["model_file"]) == row["model_sha256"], "Model checksum changed")
                require(checksum(row["feature_file"]) == row["feature_file_sha256"], "Feature file checksum changed")
                raw = frame(Path(row["ready_unlabelled"]))
                expected = frame(Path(row["ready_labelled"]))
                require(raw.equals(expected.drop(columns=["source_summary", "cluster5"])), "Ready-table predictor mismatch")
                directory = Path(row["model_file"]).parents[1]
                check_tuning(directory / "05_cv")
                predictions = check_evaluation(directory / "labelled_ready/07_test_eval", expected)
                classes = sorted(expected.source_summary.unique())
                fitted = check_artifact_probabilities(Path(row["model_file"]), raw, predictions, classes)
                check_configuration(fitted, row["model"], row["sampling"])
                from maGeneLearn.matrix_io import is_bundle, read_bundle
                if is_bundle(row["ready_unlabelled"]):
                    read_bundle(row["ready_unlabelled"])
                    read_bundle(row["ready_labelled"])
                    require(getattr(fitted, "mgl_input_format", None) == "csr", "CSR audit received a TSV-trained model")
                    require(fitted.mgl_feature_names == list(raw.columns), "CSR artifact feature order")
                    require(getattr(fitted, "mgl_resampling_policy", None) == ("none" if row["sampling"] == "none" else "native_csr"),
                            "CSR artifact omitted its native resampling policy")
                    if row["model"] == "XGBC":
                        require("dense_zeros" in fitted.named_steps, "XGBoost sparse zeros changed semantics")
                estimator = fitted.named_steps["model"] if hasattr(fitted, "named_steps") else fitted
                if row.get("lr_penalty"):
                    require(estimator.penalty == row["lr_penalty"], "Saved LR penalty differs from requested branch")
                if row.get("xgb_policy"):
                    def policies(value):
                        if isinstance(value, dict):
                            return ([value["grow_policy"]] if "grow_policy" in value else []) + [p for v in value.values() for p in policies(v)]
                        if isinstance(value, list):
                            return [p for v in value for p in policies(v)]
                        return []
                    require(set(policies(json.loads(estimator.get_booster().save_config()))) == {row["xgb_policy"]},
                            "Saved XGBoost growth policy differs from requested branch")
                importance_paths = list((directory / "06_train_eval").glob("*_feature_importances.tsv"))
                if row["model"] != "SVM":
                    require(len(importance_paths) == 1, "Missing feature importance")
                for path in importance_paths:
                    values = frame(path)
                    require(values.index.is_unique and set(values.index) == set(raw.columns), "Importance feature alignment")
                    require(np.isfinite(values.to_numpy()).all(), "Nonfinite feature importance")
                shap_paths = list((directory / "06_train_eval").glob("*_shap_values.tsv"))
                for path in shap_paths:
                    values = frame(path)
                    require(list(values.columns) == list(raw.columns), "SHAP feature order")
                    training = frame(one(directory / "06_train_eval", "*_predictions_probabilities.tsv"))
                    require(values.index.is_unique and set(values.index) == set(training.index), "SHAP sample alignment")
                    require(np.isfinite(values.to_numpy()).all(), "Nonfinite SHAP values")
                if row["model"] in {"RFC", "XGBC"} and args.profile == "full":
                    require(len(shap_paths) == len(classes), "Missing tree SHAP class exports")
                    require(all(any(path.name.endswith("_" + label + "_shap_values.tsv") for path in shap_paths)
                                for label in classes), "SHAP class names")
                shap_shape = None
                if shap_paths:
                    saved = np.load(one(directory / "06_train_eval", "*_shap_values.npy"), allow_pickle=False)
                    require(saved.ndim in {2, 3}, "Unexpected raw SHAP dimensions")
                    if saved.ndim == 3:
                        require(saved.shape[0] == len(classes), "Raw SHAP class axis")
                    for j, label in enumerate(classes):
                        values = frame(one(directory / "06_train_eval", "*_" + label + "_shap_values.tsv"))
                        require(values.index.equals(training.index), "SHAP sample order differs from evaluation")
                        np.testing.assert_allclose(saved[j] if saved.ndim == 3 else saved, values.to_numpy(), atol=1e-10)
                    shap_shape = list(saved.shape)
                result.update(status="passed", shap_files=len(shap_paths), importance_files=len(importance_paths),
                              shap_status="omitted_by_quick_profile" if args.profile == "quick" and row["model"] in {"RFC", "XGBC"} else "checked" if shap_paths else "not_applicable")
                result["shap_array_shape"] = shap_shape
            except Exception as error:
                result.update(status="failed", error=str(error))
            results.append(result)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if result["status"] == "failed" or len(results) % 10 == 0:
                print(len(results), result["status"], result.get("error", ""), flush=True)
    print("Audited", len(results), "saved models", flush=True)
    return int(any(r["status"] != "passed" for r in results))


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Targeted runtime checks supplementing, not replacing, CLI matrix validation."""
import importlib.util
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import ROOT, checksum, run_process


def load_step(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "maGeneLearn/steps" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "test")
    parser.add_argument("--output", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--only-cli", action="store_true", help="Run only fast invalid-combination checks")
    modes.add_argument("--only-orchestration", action="store_true", help="Run full train command through computed selection and hold-out evaluation")
    modes.add_argument("--only-svm-importance", action="store_true", help="Validate all predictors on smaller, documented explanation-only subsets")
    modes.add_argument("--only-data-edges", action="store_true", help="Check missing predictors, unseen outcomes, and empty Boruta selection")
    parser.add_argument("--models-run", type=Path, help="Model outputs from the current validation run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg", LOKY_MAX_CPU_COUNT="1")
    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    training = load_step("04_train_model.py", "fixture_training")
    selection = load_step("02_feature_selection.py", "fixture_selection")
    results = []

    def check(name, function):
        started = time.monotonic()
        try:
            details = function()
            result = {"check": name, "status": "passed", "details": details}
        except Exception as error:
            result = {"check": name, "status": "failed", "error": str(error)}
        result["seconds"] = time.monotonic() - started
        results.append(result)
        (args.output / "report.json").write_text(json.dumps({"python": sys.version, "executable": sys.executable,
            "manifest_sha256": checksum(args.fixtures / "manifest.json"), "results": results}, indent=2) + "\n")
        print(name, result["status"], result.get("error", ""), flush=True)

    if args.only_data_edges:
        from matrix import check_artifact_probabilities, frame, one, require
        with (args.models_run / "model_index.tsv").open() as handle:
            index = list(csv.DictReader(handle, delimiter="\t"))

        def command(argv, output):
            output.mkdir()
            with (output / "command.log").open("x") as handle:
                result = run_process(argv, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT),
                    "LOKY_MAX_CPU_COUNT": "1", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(args.output.resolve() / "matplotlib_cache")},
                    stdout=handle, stderr=subprocess.STDOUT)
            (output / "command.json").write_text(json.dumps({"command": argv, "exit_code": result.returncode}, indent=2) + "\n")
            return result.returncode, (output / "command.log").read_text()

        for problem in ["multiclass", "binary"]:
            matches = [r for r in index if r["problem"] == problem and r["model"] == "XGBC"
                       and r["selection"] == "none" and r["sampling"] == "none"]
            require(len(matches) == 1, "Need an existing no-selection XGBC model for each problem")
            choice = matches[0]
            ext = args.fixtures / "dataset3_test" / problem

            def missing_feature(problem=problem, choice=choice, ext=ext):
                original = frame(ext / "features.tsv")
                missing = next(c for c in original if original[c].nunique() > 1)
                input_path = (args.output / (problem + "_missing_feature.tsv")).resolve()
                original.drop(columns=missing).to_csv(input_path, sep="\t")
                expected = original.copy()
                expected[missing] = 0
                output = (args.output / (problem + "_missing_feature")).resolve()
                argv = [sys.executable, "-c", "from maGeneLearn.cli import cli; cli()", "test", "--name", "missing",
                        "--model-file", choice["model_file"], "--features", str(input_path), "--feature-file", choice["feature_file"],
                        "--predict-only", "--output-dir", str(output)]
                code, _ = command(argv, output)
                require(code == 0, "Missing-feature extraction failed")
                require(frame(output / "03_final_features/missing_test.tsv").equals(expected), "Missing feature was not zero-filled in source order")
                predicted = frame(one(output / "07_test_eval", "*_predictions.tsv")).rename(columns={"Prediction": "prediction"})
                classes = sorted(frame(ext / "metadata.tsv").source_summary.unique())
                check_artifact_probabilities(choice["model_file"], expected, predicted, classes)
                return {"missing_feature": missing, "zero_filled_samples": len(expected), "features": len(expected.columns)}
            check(problem + "/missing_feature_zero_fill", missing_feature)

            def unseen_label(problem=problem, choice=choice, ext=ext):
                invalid = frame(ext / "ready_labelled.tsv")
                invalid.iloc[0, invalid.columns.get_loc("source_summary")] = "UNSEEN_SOURCE_FOR_TEST"
                input_path = (args.output / (problem + "_unseen_source.tsv")).resolve()
                invalid.to_csv(input_path, sep="\t")
                output = (args.output / (problem + "_unseen_source")).resolve()
                argv = [sys.executable, "-c", "from maGeneLearn.cli import cli; cli()", "test", "--name", "unseen",
                        "--model-file", choice["model_file"], "--features-test", str(input_path), "--label", "source_summary",
                        "--group-column", "cluster5", "--output-dir", str(output)]
                code, log = command(argv, output)
                require(code != 0 and "UNSEEN_SOURCE_FOR_TEST" in log and "unseen labels" in log,
                        "Unknown source was not explicitly rejected")
                return {"expected_rejection_exit_code": code, "unseen_source": "UNSEEN_SOURCE_FOR_TEST"}
            check(problem + "/unseen_source_rejected", unseen_label)

        base = args.fixtures / "dataset2_binary"
        metadata = frame(base / "train_metadata.tsv")
        names = frame(base / "features.tsv").columns[:16]
        constant_path = (args.output / "synthetic_zero_candidates.tsv").resolve()
        pd.DataFrame(0, index=metadata.index, columns=names, dtype="int8").to_csv(constant_path, sep="\t")
        for feature_model in ["RFC", "XGBC"]:
            def empty_selection(feature_model=feature_model):
                output = (args.output / ("empty_boruta_" + feature_model)).resolve()
                argv = [sys.executable, str(ROOT / "maGeneLearn/steps/02_feature_selection.py"), "--method", "boruta",
                        "--model", feature_model, "--train_data", str((base / "train_metadata.tsv").resolve()),
                        "--chisq_file", str(constant_path), "--group_col", "cluster5", "--outcome_col", "source_summary",
                        "--filtered_train_dir", str(output / "filtered"), "--output", str(output), "--name", "empty",
                        "--n-jobs", "1", "--max-iter", "100"]
                code, log = command(argv, output)
                require(code != 0 and "Boruta rejected all features" in log and not list(output.glob("*_min.tsv")),
                        "Empty Boruta selection was not clearly rejected")
                return {"expected_rejection_exit_code": code, "synthetic_input_sha256": checksum(constant_path),
                        "synthetic_input": "16 all-zero candidate columns; original sample labels/groups retained; originals unchanged"}
            check("empty_boruta/" + feature_model, empty_selection)
        return int(any(r["status"] != "passed" for r in results))

    if args.only_svm_importance:
        from matrix import check_evaluation, frame, one, require
        with (args.models_run / "model_index.tsv").open() as handle:
            index = list(csv.DictReader(handle, delimiter="\t"))
        for problem, dataset in [("multiclass", "dataset1_multiclass"), ("binary", "dataset2_binary")]:
            def importance(problem=problem, dataset=dataset):
                choices = [r for r in index if r["problem"] == problem and r["model"] == "SVM"
                           and r["selection"] == "none" and r["sampling"] == "none"]
                require(len(choices) == 1, "Need one successful no-selection SVM model per outcome type")
                choice = choices[0]
                require(checksum(choice["model_file"]) == choice["model_sha256"], "SVM model checksum changed")
                base = args.fixtures / dataset
                metadata = frame(base / "train_metadata.tsv")
                ready = frame(base / "ready_train.tsv")
                ids = []
                for source in sorted(metadata.source_summary.unique()):
                    for lineage in ["19", "34"]:
                        candidates = metadata.loc[(metadata.source_summary == source) & (metadata.plot_MLST.astype(str) == lineage)]
                        require(len(candidates) >= 6, "Insufficient diagnostic source/lineage samples")
                        ids.extend(candidates.index[:6])
                small = ready.loc[ids]
                predictors = small.drop(columns=["source_summary", "cluster5"])
                require(predictors.shape[1] == 2471, "SVM diagnostic must retain every predictor")
                for train_indices, test_indices in training.get_cv_splits(predictors, small.source_summary.to_numpy(), small.cluster5.to_numpy(), 3):
                    require(not set(small.iloc[train_indices].cluster5) & set(small.iloc[test_indices].cluster5), "Diagnostic group overlap")
                    require(small.iloc[train_indices].source_summary.value_counts().min() >= 5, "Insufficient calibration samples")
                input_path = (args.output / (problem + "_inputs.tsv")).resolve()
                small.to_csv(input_path, sep="\t")
                output = (args.output / problem).resolve()
                output.mkdir()
                argv = [sys.executable, str(ROOT / "maGeneLearn/steps/05_evaluate_model.py"),
                        "--model", choice["model_file"], "--features", str(input_path), "--n_splits", "3",
                        "--label", "source_summary", "--group_column", "cluster5", "--scoring", "balanced_accuracy",
                        "--name", "importance_" + problem, "--output_dir", str(output)]
                env = {**os.environ, "LOKY_MAX_CPU_COUNT": "1", "MPLBACKEND": "Agg",
                       "MPLCONFIGDIR": str(args.output.resolve() / "matplotlib_cache")}
                with (output / "command.log").open("x") as handle:
                    completed = run_process(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=7200)
                (output / "command.json").write_text(json.dumps({"command": argv, "exit_code": completed.returncode,
                                                               "LOKY_MAX_CPU_COUNT": "1"}, indent=2) + "\n")
                require(completed.returncode == 0, "SVM importance failed; see " + str(output / "command.log"))
                check_evaluation(output, small)
                values = frame(one(output, "*_feature_importances.tsv"))
                require(values.index.is_unique and set(values.index) == set(predictors.columns), "SVM importance feature coverage")
                require(np.isfinite(values.to_numpy()).all(), "Nonfinite SVM importance")
                np.testing.assert_allclose(values["average"], values.drop(columns="average").mean(axis=1), atol=1e-10)
                return {"samples": len(small), "sample_ids": ids, "features": len(values), "cv_folds": 3,
                        "permutation_repeats": 10, "source_model": choice["model_file"], "input_sha256": checksum(input_path),
                        "diagnostic_only": "Smaller training-only subset; full main datasets unchanged"}
            check("svm_full_feature_importance/" + problem, importance)
        return int(any(r["status"] != "passed" for r in results))

    if args.only_orchestration:
        from matrix import check_evaluation, frame, one, require
        for dataset in ["dataset1_multiclass", "dataset2_binary"]:
            directory = args.fixtures / dataset
            for route in ["automatic", "supplied"]:
                def workflow(dataset=dataset, directory=directory, route=route):
                    name = dataset + "_" + route
                    output = (args.output / name).resolve()
                    output.mkdir()
                    argv = [sys.executable, "-c", "from maGeneLearn.cli import cli; cli()", "train",
                            "--name", name, "--model", "LR", "--feature-model", "RFC", "--chisq", "--muvr",
                            "--features", str((directory / "features.tsv").resolve()), "--output-dir", str(output),
                            "--label", "source_summary", "--group-column", "cluster5", "--id-col", "Sample.ID",
                            "--lineage-col", "plot_MLST", "--n-splits", "5", "--n-splits-cv", "3",
                            "--n-jobs", "1", "--n-iter", "3", "--k", "256", "--muvr-n-repetitions", "1",
                            "--muvr-n-outer", "3", "--muvr-n-inner", "3"]
                    if route == "automatic":
                        argv += ["--meta-file", str((directory / "metadata.tsv").resolve())]
                    else:
                        argv += ["--train-meta", str((directory / "train_metadata.tsv").resolve()),
                                 "--test-meta", str((directory / "holdout_metadata.tsv").resolve())]
                    with (output / "command.log").open("x") as log:
                        completed = run_process(argv, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT),
                            "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(args.output.resolve() / "matplotlib_cache")}, stdout=log, stderr=subprocess.STDOUT)
                    (output / "command.json").write_text(json.dumps({"command": argv, "exit_code": completed.returncode}, indent=2) + "\n")
                    require(completed.returncode == 0, "Full training failed; see " + str(output / "command.log"))
                    one(output / "04_model", "*.joblib")
                    train = frame(output / "03_final_features" / (name + "_train.tsv"))
                    holdout = frame(output / "03_final_features" / (name + "_test.tsv"))
                    require(not set(train.cluster5) & set(holdout.cluster5), "Workflow group overlap")
                    require(set(train.index) | set(holdout.index) == set(frame(directory / "metadata.tsv").index), "Workflow sample coverage")
                    check_evaluation(output / "06_train_eval", train)
                    require((output / "07_test_eval").is_dir(), "CLI exited successfully but omitted 07_test_eval hold-out evaluation: " + str(output))
                    check_evaluation(output / "07_test_eval", holdout)
                    return {"training": len(train), "holdout": len(holdout), "command": argv}
                check(dataset + "/full_train/" + route, workflow)
        return int(any(r["status"] != "passed" for r in results))

    def float_precision():
        from matrix import frame
        from sklearn.metrics import average_precision_score
        path = args.output / "float_precision.tsv"
        path.write_text("Sample.ID\tscore\none\t0.1355125536124339\ntwo\t0.13551255361243392\n")
        scores = frame(path).score
        np.testing.assert_array_equal(scores, [0.1355125536124339, 0.13551255361243392])
        if average_precision_score([0, 1], scores) != 1.0:
            raise ValueError("Probability parser changed score ranks/ties")
        return {"average_precision": 1.0, "distinct_scores": int(scores.nunique())}
    check("probability_reader_preserves_rank_ties", float_precision)

    directory = args.fixtures / "dataset2_binary"
    base = ["train", "--train-meta", str(directory / "train_metadata.tsv"), "--features", str(directory / "features.tsv"),
            "--feature-selection-only", "--name", "invalid", "--label", "source_summary", "--group-column", "cluster5"]
    cases = [
        ("missing_metadata", ["train", "--name", "invalid"], "Supply EITHER --meta-file"),
        ("muvr_without_chisq", base + ["--muvr"], "requires --chisq or --chisq-file"),
        ("simultaneous_selectors", base + ["--chisq", "--muvr", "--boruta"], "Choose either --muvr or --boruta"),
        ("conflicting_chisq_routes", base + ["--chisq", "--chisq-file", str(directory / "all_features.tsv")], "mutually exclusive"),
        ("chisq_without_selector", base + ["--chisq-file", str(directory / "all_features.tsv")], "Chi² without MUVR/Boruta not supported"),
        ("invalid_feature_model", base + ["--feature-model", "LR"], "is not one of"),
    ]
    for name, arguments, message in cases:
        def invalid_check(name=name, arguments=arguments, message=message):
            argv = [sys.executable, "-c", "from maGeneLearn.cli import cli; cli()"] + arguments + ["--output-dir", str(args.output / name)]
            result = run_process(argv, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
            (args.output / (name + ".log")).write_text(result.stdout + result.stderr)
            if result.returncode == 0 or message not in result.stdout + result.stderr:
                raise ValueError("Expected clear CLI rejection: " + message)
            return {"command": argv, "expected_rejection_exit_code": result.returncode}
        check("cli/" + name, invalid_check)
    if args.only_cli:
        return int(any(r["status"] != "passed" for r in results))

    def dedup():
        path = args.fixtures / "selector_dedup_case"
        data = selection.prepare_data_muvr(str(path / "metadata.tsv"), str(args.output / "dedup"),
                                           "dedup", "cluster5", "source_summary")
        expected = pd.read_csv(path / "expected_retained.tsv", sep="\t")["Sample.ID"]
        np.testing.assert_array_equal(data.index, expected)
        return {"input_rows": 6, "retained_rows": len(data)}
    check("selector_last_sorted_dedup", dedup)

    for dataset in ["dataset1_multiclass", "dataset2_binary"]:
        data = pd.read_csv(args.fixtures / dataset / "ready_train.tsv", sep="\t", index_col=0)
        X = data.drop(columns=["source_summary", "cluster5"])
        y = LabelEncoder().fit_transform(data.source_summary)
        groups = data.cluster5.to_numpy()
        folds = training.get_cv_splits(X, y, groups, 3)
        for sampler in ["random", "smote", "random_under", "enn", "smoteenn"]:
            def sampling_check(sampler=sampler):
                counts = []
                for neighbors in ([3, 10] if sampler in {"smote", "enn", "smoteenn"} else [None]):
                    for indices, _ in folds:
                        pipeline = training.prepare_pipeline("LR", sampler, 1)
                        if sampler == "smote":
                            pipeline.set_params(sampler__k_neighbors=neighbors)
                        elif sampler == "enn":
                            pipeline.set_params(sampler__n_neighbors=neighbors)
                        elif sampler == "smoteenn":
                            pipeline.set_params(sampler__smote__k_neighbors=neighbors, sampler__enn__n_neighbors=neighbors)
                        sampled_X, sampled_y = pipeline.named_steps["sampler"].fit_resample(X.iloc[indices], y[indices])
                        if set(sampled_y) != set(y):
                            raise ValueError("Sampler removed an entire class")
                        counts.append({"neighbors": neighbors, "classes": np.bincount(sampled_y).tolist(), "rows": len(sampled_X)})
                return counts
            check(dataset + "/sampler/" + sampler, sampling_check)

        for kernel in ["linear", "rbf"]:
            def svm_check(kernel=kernel):
                trial = optuna.trial.FixedTrial({"model__C": 1.0, "model__kernel": kernel, "model__gamma": "scale",
                                                "model__class_weight": None})
                score, details = training.optuna_objective(trial, training.prepare_pipeline("SVM", "none", 1),
                    X, y, groups, folds, ["balanced_accuracy"], "SVM", "none", 1)
                if not np.isfinite(score):
                    raise ValueError("Nonfinite SVM score")
                return {"kernel": kernel, "score": float(score), "folds": len(folds)}
            check(dataset + "/svm/" + kernel, svm_check)

        for penalty in ["l1", "l2", "elasticnet"]:
            def lr_check(penalty=penalty):
                trial = optuna.trial.FixedTrial({"model__C": 1.0, "model__class_weight": None, "model__l1_ratio": 0.5})
                score, _ = training.optuna_objective(trial, training.prepare_pipeline("LR", "none", 1, lr_penalty=penalty),
                    X, y, groups, folds, ["balanced_accuracy"], "LR", "none", 1)
                if not np.isfinite(score):
                    raise ValueError("Nonfinite LR score")
                return {"penalty": penalty, "score": float(score), "folds": len(folds)}
            check(dataset + "/lr/" + penalty, lr_check)

        for policy in ["depthwise", "lossguide"]:
            def xgb_check(policy=policy):
                params = {"model__n_estimators": 200, "model__learning_rate": 0.1, "model__max_depth": 3,
                          "model__max_leaves": 16, "model__min_child_weight": 1, "model__gamma": 0,
                          "model__subsample": 1.0, "model__colsample_bytree": 1.0,
                          "model__reg_alpha": 1e-8, "model__reg_lambda": 1.0}
                trial = optuna.trial.FixedTrial(params)
                score, _ = training.optuna_objective(trial, training.prepare_pipeline("XGBC", "none", 1, xgb_policy=policy),
                    X, y, groups, folds, ["balanced_accuracy"], "XGBC", "none", 1)
                if not np.isfinite(score):
                    raise ValueError("Nonfinite XGBC score")
                return {"policy": policy, "score": float(score), "folds": len(folds)}
            check(dataset + "/xgb/" + policy, xgb_check)

    return int(any(r["status"] != "passed" for r in results))


if __name__ == "__main__":
    sys.exit(main())

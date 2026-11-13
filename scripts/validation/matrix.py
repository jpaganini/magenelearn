#!/usr/bin/env python3
"""Serial fixture integration runner. Requires the setup.py dependencies."""
import argparse
import collections
import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

from common import ROOT, checksum, run_process

MODELS = ["RFC", "XGBC", "SVM", "LR"]
SELECTIONS = ["none", "muvr_RFC", "muvr_XGBC", "boruta_RFC", "boruta_XGBC"]
SAMPLINGS = ["none", "random", "smote", "random_under", "enn", "smoteenn"]
ENN_REJECTION_CASE = ("multiclass", "muvr_RFC", "XGBC", "enn")
ENN_REJECTION_POLICY = "multiclass_muvr_RFC_XGBC_enn_class_loss_v1"


def eligible_enn_rejection(result):
    return (result.get("kind") == "model" and
            tuple(result.get(k) for k in ("dataset", "selection", "model", "sampling")) == ENN_REJECTION_CASE)


def unexpected_rejection_outputs(directory):
    """A training rejection must not have reached evaluation or saved a model."""
    forbidden = [p for p in (directory / "04_model").rglob("*")
                 if p.is_file() and p.suffix != ".log"]
    for name in ("06_train_eval", "07_test_eval", "external_inputs", "labelled_ready",
                 "labelled_extraction", "predict_ready", "predict_extraction"):
        forbidden.extend(p for p in (directory / name).rglob("*") if p.is_file())
    forbidden.extend(directory.rglob("*.joblib"))
    return sorted({str(p.relative_to(directory)) for p in forbidden})


def expected_rejection_errors(result, check_files=True):
    """Validate the narrow policy and its evidence; a claimed status is insufficient."""
    try:
        require(eligible_enn_rejection(result), "Rejection is outside the approved case")
        evidence = result["rejection"]
        require(evidence["policy"] == ENN_REJECTION_POLICY, "Unknown rejection policy")
        require(evidence["exit_code"] == 1, "Expected a training exit code of one")
        requested = evidence["requested_neighbors"]
        require(type(requested) is int and 3 <= requested <= 10, "Invalid requested neighbour count")
        require(type(evidence["trial"]) is int and 0 <= evidence["trial"] < 3, "Invalid optimization trial")
        require(evidence["seed"] == 50 and evidence["cv_folds"] == 3, "Wrong CV configuration")
        require(evidence["sampler"] == {"kind_sel": "mode", "sampling_strategy": "auto"}, "Wrong ENN configuration")
        classes = evidence["classes"]
        require(classes == ["Bovine", "Poultry", "Swine"], "Wrong class encoding")
        ids, features = evidence["sample_ids"], evidence["features"]
        require(bool(ids) and len(ids) == len(set(ids)), "Missing or duplicate sample IDs")
        require(bool(features) and len(features) == len(set(features)), "Missing or duplicate predictors")
        partitions = evidence["partitions"]
        names = ["fold_1", "fold_2", "fold_3", "final_fit"]
        require([p["name"] for p in partitions] == names, "Missing training partitions")
        for part in partitions:
            before = part["before"]
            require(set(before) == set(classes) and all(type(n) is int and n > 0 for n in before.values()),
                    "A class was already absent before sampling")
            require(sum(before.values()) == len(part["sample_ids"]), "Partition sample counts differ")
            require(len(part["sample_ids"]) == len(set(part["sample_ids"])) and
                    set(part["sample_ids"]) <= set(ids), "Invalid partition sample IDs")
            require(len(part["groups"]) == len(part["sample_ids"]), "Partition group alignment")
            if part["name"] != "final_fit":
                require(not set(part["groups"]) & set(part["validation_groups"]), "CV group leakage")
                require(not set(part["sample_ids"]) & set(part["validation_ids"]) and
                        set(part["sample_ids"]) | set(part["validation_ids"]) == set(ids), "CV sample coverage")
        require(partitions[-1]["sample_ids"] == ids, "Final-fit sample order differs")
        candidates = evidence["candidates"]
        require([c["neighbors"] for c in candidates] == list(range(requested, 0, -1)), "Missing ENN candidates")
        for candidate in candidates:
            require([c["partition"] for c in candidate["counts"]] == names, "Missing candidate partition counts")
            loses_class = False
            for counts, part in zip(candidate["counts"], partitions):
                after = counts["after"]
                require(set(after) == set(classes) and all(type(n) is int and 0 <= n <= part["before"][label]
                                                         for label, n in after.items()), "Invalid post-ENN counts")
                loses_class |= any(n == 0 for n in after.values())
            require(loses_class, "An ENN candidate preserves every class")
        require(evidence["outputs_absent"] is True, "Model or evaluation output was produced")
        for key in ("training_table", "command_log", "training_log"):
            item = evidence[key]
            require(bool(item["path"]) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
                    "Missing input/log checksum")
            if check_files:
                require(checksum(Path(item["path"])) == item["sha256"], "Input/log evidence changed")
        require(bool(evidence["command"]), "Missing training command")
        if check_files:
            require(not unexpected_rejection_outputs(Path(evidence["case_directory"])), "Rejection has model/evaluation outputs")
        if result.get("status") == "expected_rejection":
            require(bool(result["evidence_file"]) and
                    re.fullmatch(r"[0-9a-f]{64}", result["evidence_sha256"]) is not None, "Missing verification report")
            if check_files:
                path = Path(result["evidence_file"])
                require(checksum(path) == result["evidence_sha256"] and
                        json.loads(path.read_text()) == evidence, "Verification report changed")
        return []
    except (KeyError, TypeError, ValueError, OSError, AttributeError, IndexError) as error:
        return [str(error)]


def verify_enn_rejection(result, train, directory, command):
    """Independently reproduce exhausted ENN using current inputs, never the production helper."""
    import numpy as np
    from imblearn.under_sampling import EditedNearestNeighbours
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder

    require(eligible_enn_rejection(result), "Rejection is outside the approved case")
    require(command.get("exit_code") == 1 and command.get("status") != "timed_out", "Not a training failure")
    require(Path(command["log"]).resolve() == (directory / "command.log").resolve(), "Failure occurred after training")
    argv = command["command"]
    require("--features-train" in argv and
            Path(argv[argv.index("--features-train") + 1]).resolve() == train.resolve(), "Wrong training input")
    command_log, training_log = directory / "command.log", directory / "04_model/train.log"
    require("Step 'train' failed with exit code 1." in command_log.read_text(), "Failure was not in the training step")
    pattern = (r"^(?:maGeneLearn\.steps\.)?sampling_validation\.SamplingClassLoss: "
               r"Optimization trial (\d+): no ENN neighbour count from (\d+) to 1 "
               r"preserved every class\. Sampling was not disabled\.$")
    training_text = training_log.read_text()
    matches = set(re.findall(pattern, training_text, re.MULTILINE))
    require(len(matches) == 1, "No unambiguous exhausted-ENN traceback")
    require(re.fullmatch(pattern, training_text.rstrip().splitlines()[-1]) is not None,
            "The terminal training error is not exhausted ENN")
    trial, requested = map(int, next(iter(matches)))
    require(0 <= trial < 3 and 3 <= requested <= 10, "Unexpected optimization parameters")
    require(not unexpected_rejection_outputs(directory), "Training produced model/evaluation outputs")
    data = frame(train)
    require(data.index.is_unique and data.columns.is_unique, "Duplicate training identifiers")
    X = data.drop(columns=["source_summary", "cluster5"])
    encoder = LabelEncoder()
    y = encoder.fit_transform(data.source_summary)
    classes = encoder.classes_.tolist()
    require(classes == ["Bovine", "Poultry", "Swine"], "Wrong training labels")
    groups = data.cluster5.to_numpy()
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=50).split(X, y, groups))
    rows = [("fold_%d" % (i + 1), tr, va) for i, (tr, va) in enumerate(splits)]
    rows.append(("final_fit", np.arange(len(y)), np.array([], dtype=int)))

    def counts(values):
        return {label: int(np.sum(values == i)) for i, label in enumerate(classes)}

    evidence = {"policy": ENN_REJECTION_POLICY, "exit_code": 1, "trial": trial,
                "requested_neighbors": requested, "seed": 50, "cv_folds": 3,
                "sampler": {"kind_sel": "mode", "sampling_strategy": "auto"}, "classes": classes,
                "sample_ids": data.index.astype(str).tolist(), "features": X.columns.tolist(),
                "case_directory": str(directory), "command": argv, "outputs_absent": True,
                "partitions": [], "candidates": []}
    for name, tr, va in rows:
        evidence["partitions"].append({"name": name, "sample_ids": data.index[tr].astype(str).tolist(),
                                        "groups": groups[tr].astype(str).tolist(), "before": counts(y[tr]),
                                        "validation_ids": data.index[va].astype(str).tolist(),
                                        "validation_groups": groups[va].astype(str).tolist()})
    for neighbors in range(requested, 0, -1):
        candidate = {"neighbors": neighbors, "counts": []}
        for name, tr, _ in rows:
            _, sampled = EditedNearestNeighbours(kind_sel="mode", n_neighbors=neighbors).fit_resample(X.iloc[tr], y[tr])
            candidate["counts"].append({"partition": name, "after": counts(sampled)})
        evidence["candidates"].append(candidate)
    for key, path in (("training_table", train), ("command_log", command_log), ("training_log", training_log)):
        evidence[key] = {"path": str(path), "sha256": checksum(path)}
    errors = expected_rejection_errors({**result, "rejection": evidence})
    require(not errors, "; ".join(errors))
    return evidence


def require(condition, message):
    if not condition:
        raise ValueError(message)


def one(directory, pattern):
    matches = list(directory.glob(pattern))
    if not matches and pattern.endswith(".tsv"):
        matches = list(directory.glob(pattern[:-4] + ".csr/manifest.json"))
    require(len(matches) == 1, "Expected one %s in %s, got %d" % (pattern, directory, len(matches)))
    return matches[0]


def frame(path):
    import pandas as pd
    from common import matrix_path
    from maGeneLearn.matrix_io import is_bundle, read_table
    path = matrix_path(path)
    if is_bundle(path):
        table = read_table(path)
        for column in table:
            if isinstance(table[column].dtype, pd.SparseDtype):
                values = table[column].sparse.to_dense()
                table[column] = values.astype("int64") if values.dtype.kind in "biu" else values
            elif column in {"cluster5", "plot_MLST"}:
                table[column] = pd.to_numeric(table[column], errors="ignore")
        return table
    # AP depends on score ranks/ties, including distinctions of one float64 ULP.
    return pd.read_csv(path, sep="\t", index_col=0, float_precision="round_trip")


def export_model_index(report, destination):
    """Pair successful saved models with compatible ready tables and feature lists."""
    fixtures = Path(report["arguments"]["fixtures"])
    selectors = {(r["dataset"], r["selection"]): r["exports"][0] for r in report["results"]
                 if r["kind"] == "selector" and r["status"] == "passed"}
    columns = ["problem", "model", "selection", "sampling", "model_file", "model_sha256", "feature_file",
               "feature_file_sha256", "ready_labelled", "ready_unlabelled", "raw_features", "test_metadata", "lr_penalty", "xgb_policy"]
    with destination.open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        for result in report["results"]:
            if result["kind"] != "model" or result["status"] != "passed":
                continue
            problem = result["dataset"]
            artifact = Path(result["artifact"])
            case = artifact.parents[1]
            dataset = "dataset1_multiclass" if problem == "multiclass" else "dataset2_binary"
            selected = fixtures / dataset / "all_features.tsv" if result["selection"] == "none" else Path(selectors[(problem, result["selection"])])
            external = fixtures / "dataset3_test" / problem
            paths = [artifact, selected, case / "external_inputs/ready.tsv", case / "external_inputs/predict.tsv",
                     external / "features.tsv", external / "metadata.tsv"]
            from common import matrix_path
            paths = [matrix_path(p) for p in paths]
            if report["arguments"].get("input_format") == "csr":
                for i, path in enumerate(paths):
                    if i in {0, 5}:
                        continue
                    converted = destination.parent / "csr_inputs" / checksum(path) / "manifest.json"
                    if converted.is_file():
                        paths[i] = converted
            selected = paths[1]
            require(all(p.is_file() for p in paths), "Incomplete saved-model pairing")
            writer.writerow([problem, result["model"], result["selection"], result["sampling"], str(artifact), checksum(artifact),
                             str(selected), checksum(selected), *map(str, paths[2:]),
                             report["arguments"]["lr_penalty"] if result["model"] == "LR" else "",
                             report["arguments"]["xgb_policy"] if result["model"] == "XGBC" else ""])


def check_artifact_probabilities(artifact, raw, predictions, classes):
    """Check column values directly against fitted classes, independently of evaluation helpers."""
    import joblib
    import numpy as np
    import pandas as pd
    loaded = joblib.load(artifact)
    model = loaded["model"] if isinstance(loaded, dict) else loaded
    encoder = loaded.get("label_encoder") if isinstance(loaded, dict) else None
    fitted_classes = np.asarray(model.classes_)
    labels = model.predict(raw)
    if encoder is not None:
        labels = encoder.inverse_transform(labels.astype(int))
        if np.issubdtype(fitted_classes.dtype, np.integer):
            fitted_classes = encoder.inverse_transform(fitted_classes.astype(int))
    require(set(fitted_classes) <= set(classes), "Unknown fitted class")
    np.testing.assert_array_equal(predictions.loc[raw.index, "prediction"], labels)
    direct = pd.DataFrame(model.predict_proba(raw), index=raw.index, columns=fitted_classes)
    direct = direct.reindex(columns=classes, fill_value=0)
    np.testing.assert_allclose(predictions.loc[raw.index, classes], direct, atol=1e-6, rtol=1e-6)
    return model


def check_evaluation(directory, expected):
    """Recompute classification, confusion, MCC and AP from emitted predictions."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, matthews_corrcoef
    predictions = frame(one(directory, "*_predictions_probabilities.tsv"))
    classes = sorted(expected["source_summary"].unique())
    require(predictions.index.is_unique and set(predictions.index) == set(expected.index), "Prediction sample coverage")
    require(predictions["truth"].equals(expected.loc[predictions.index, "source_summary"].rename("truth")), "Truth alignment")
    require(set(predictions["prediction"]) <= set(classes), "Invalid predicted label")
    require(list(predictions.columns) == classes + ["truth", "prediction"], "Probability/class order")
    probabilities = predictions[classes].to_numpy()
    require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all(), "Probability range")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    cm = frame(one(directory, "*_confusion_matrix.tsv"))
    require(list(cm.index) == classes and list(cm.columns) == classes, "Confusion class order")
    np.testing.assert_array_equal(cm.to_numpy(), confusion_matrix(predictions.truth, predictions.prediction, labels=classes))
    require(cm.to_numpy().sum() == len(expected), "Confusion total")
    report = pd.read_csv(one(directory, "*_classification_report.tsv"), sep="\t", index_col=0)
    independent = classification_report(predictions.truth, predictions.prediction, output_dict=True, zero_division=0)
    for label in classes + ["macro avg", "weighted avg"]:
        for metric in ["precision", "recall", "f1-score", "support"]:
            np.testing.assert_allclose(report.loc[label, metric], independent[label][metric], atol=1e-10)
    np.testing.assert_allclose(report.loc["accuracy", "f1-score"], independent["accuracy"], atol=1e-10)
    truth = np.column_stack([(predictions.truth == label).astype(int) for label in classes])
    ap = [average_precision_score(truth[:, j], probabilities[:, j]) for j in range(len(classes))]
    metrics = pd.read_csv(one(directory, "*_mcc_auprc.tsv"), sep="\t").iloc[0]
    independent_metrics = {"MCC": matthews_corrcoef(predictions.truth, predictions.prediction),
                           "Macro_AUPRC": np.mean(ap), "Micro_AUPRC": average_precision_score(truth.ravel(), probabilities.ravel())}
    independent_metrics.update({"AUPRC_" + label: ap[j] for j, label in enumerate(classes)})
    for metric, value in independent_metrics.items():
        np.testing.assert_allclose(metrics[metric], round(value, 4), atol=1e-10)
    return predictions


class Runner:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "LOKY_MAX_CPU_COUNT": "1", "MPLBACKEND": "Agg",
                    "MPLCONFIGDIR": str(self.output / "matplotlib_cache")}
        # Also constrain direct save/load inference in this process, before numpy imports.
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]:
            os.environ[key] = self.env[key]
        self.results = []
        packages = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "joblib",
                    "shap", "py-muvr", "optuna", "boruta", "click", "matplotlib"]
        self.report = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                       "versions": {p: importlib.metadata.version(p) for p in packages},
                       "concurrency": {k: self.env[k] for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]},
                       "fixture_manifest_sha256": checksum(args.fixtures / "manifest.json"),
                       "runner_sha256": checksum(Path(__file__)),
                       "expected_model_cases": len(args.datasets) * len(args.selections) * len(args.samplings) * len(args.models),
                       "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       "results": self.results}
        self.save()

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def command(self, argv, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = {"command": argv, "log": str(log)}
        self.report.setdefault("commands", []).append(entry)
        self.save()
        started = time.monotonic()
        try:
            with log.open("x") as handle:
                result = run_process(argv, cwd=ROOT, env=self.env, stdout=handle, stderr=subprocess.STDOUT)
            entry["exit_code"] = result.returncode
        except subprocess.TimeoutExpired:
            entry["exit_code"] = None
            entry["status"] = "timed_out"
            raise
        finally:
            entry["seconds"] = time.monotonic() - started
            self.save()
        require(result.returncode == 0, "Command exited %d; see %s" % (result.returncode, log))

    def cli(self, *args):
        return [sys.executable, "-c", "from maGeneLearn.cli import cli; cli()"] + [str(a) for a in args]

    def common(self):
        return ["--label", "source_summary", "--group-column", "cluster5"]

    def preflight(self, dataset, directory):
        import numpy as np
        from sklearn.model_selection import StratifiedGroupKFold
        data = frame(directory / "ready_train.tsv")
        audits = []
        for train, validation in StratifiedGroupKFold(3, shuffle=True, random_state=50).split(
                data, data.source_summary, data.cluster5):
            require(not set(data.iloc[train].cluster5) & set(data.iloc[validation].cluster5), "CV group overlap")
            counts = data.iloc[train].source_summary.value_counts()
            require(len(counts) == data.source_summary.nunique() and counts.min() >= 11, "Insufficient CV class counts for k=10")
            require(set(data.iloc[validation].source_summary) == set(data.source_summary), "Missing validation class")
            audits.append({"train_ids": data.index[train].tolist(), "validation_ids": data.index[validation].tolist(),
                           "train_class_counts": {str(k): int(v) for k, v in counts.items()}})
        self.report.setdefault("cv_folds", {})[dataset] = audits
        automatic = self.output / dataset / "automatic_split"
        self.command([sys.executable, str(ROOT / "maGeneLearn/steps/00_split_dataset.py"),
                      "--meta-file", str(directory / "metadata.tsv"), "--out-dir", str(automatic), "--name", "automatic",
                      "--id-col", "Sample.ID", "--lineage-col", "plot_MLST", "--group-col", "cluster5",
                      "--outcome-col", "source_summary", "--n-splits", "5", "--seed", "42"], automatic / "command.log")
        train, test = frame(automatic / "automatic_train.tsv"), frame(automatic / "automatic_test.tsv")
        require(not set(train.cluster5) & set(test.cluster5), "Automatic split group overlap")
        require(set(train.index) | set(test.index) == set(frame(directory / "metadata.tsv").index), "Automatic split coverage")
        require(set(train.source_summary) == set(test.source_summary) == set(data.source_summary), "Automatic split class coverage")
        self.report.setdefault("automatic_splits", {})[dataset] = {"train": len(train), "holdout": len(test)}
        self.save()

    def selector(self, dataset, directory, selection, supplied_chisq=None):
        method, feature_model = selection.split("_")
        output = self.output / dataset / "selectors" / selection
        name = "selection_" + selection
        argv = self.cli("train", "--train-meta", directory / "train_metadata.tsv", "--test-meta", directory / "holdout_metadata.tsv",
                        "--features", directory / "features.tsv", "--name", name, "--feature-model", feature_model,
                        "--feature-selection-only", "--" + method, "--output-dir", output,
                        "--id-col", "Sample.ID", "--lineage-col", "plot_MLST", "--n-jobs", "1", "--k", "256",
                        "--muvr-n-repetitions", "1", "--muvr-n-outer", "3", "--muvr-n-inner", "3",
                        "--boruta-max-iter", "100", *self.common())
        argv += ["--chisq-file", str(supplied_chisq)] if supplied_chisq else ["--chisq"]
        self.command(argv, output / "command.log")
        original = frame(directory / "features.tsv")
        labels = frame(directory / "train_metadata.tsv").source_summary
        exports = []
        for level in (["min", "mid", "max"] if method == "muvr" else ["min"]):
            path = output / ("02_" + method) / (name + "_" + selection + "_" + level + ".tsv")
            exported = frame(path)
            names = [c for c in exported if c != "source_summary"]
            require(bool(names) and len(names) == len(set(names)), "Empty/duplicate selector output")
            require(exported.index.is_unique and set(exported.index) == set(labels.index), "Selector sample coverage")
            require(set(names) <= set(original.columns), "Unknown selected features")
            require(exported[names].equals(original.loc[exported.index, names]), "Selector changed feature values")
            exports.append(str(path))
        ready = output / "03_final_features"
        selected = frame(Path(exports[0])).columns.tolist()
        selected = [c for c in selected if c != "source_summary"]
        for part, suffix in [("train", "train"), ("holdout", "test")]:
            table = frame(ready / (name + "_" + suffix + ".tsv"))
            metadata = frame(directory / (part + "_metadata.tsv"))
            require(table.index.is_unique and set(table.index) == set(metadata.index), "Extracted split coverage")
            require(list(table.drop(columns=["source_summary", "cluster5"]).columns) == selected, "Extraction feature order")
            require(table.source_summary.equals(metadata.loc[table.index].source_summary), "Extraction label alignment")
            require(table.cluster5.equals(metadata.loc[table.index].cluster5), "Extraction group alignment")
            require(table[selected].equals(original.loc[table.index, selected]), "Extraction value changes")
        chisq = supplied_chisq or one(output / "01_chisq", "*_top256_features.tsv")
        chi_table = frame(chisq)
        require(chi_table.index.is_unique and set(chi_table.index) == set(labels.index), "Chi-square training membership")
        chi_columns = chi_table.columns.tolist()
        require(chi_table.equals(original.loc[chi_table.index, chi_columns]), "Chi-square changed training values")
        require(chi_columns == [c for c in original if c in set(chi_columns)], "Chi-square source feature order")
        require(len(chi_columns) == 256, "Chi-square width")
        paths = [Path(p) for p in exports] + [ready / (name + "_train.tsv"), ready / (name + "_test.tsv"), chisq]
        return {"train": ready / (name + "_train.tsv"), "holdout": ready / (name + "_test.tsv"),
                "feature_file": Path(exports[0]), "chisq": chisq, "exports": exports,
                "checksums": {str(p): checksum(p) for p in paths}}

    def case(self, dataset, directory, selection, cached, model, sampling):
        import joblib
        import numpy as np
        import pandas as pd
        for path, expected in cached.get("checksums", {}).items():
            require(checksum(path) == expected, "Validated selection artifact changed: " + path)
        name = "%s_%s_%s_%s" % (dataset, selection, model, sampling)
        output = self.output / dataset / "models" / name
        argv = self.cli("train", "--train-meta", directory / "train_metadata.tsv", "--test-meta", directory / "holdout_metadata.tsv",
                        "--features-train", cached["train"], "--features-test", cached["holdout"], "--name", name,
                        "--model", model, "--upsampling", sampling, "--output-dir", output,
                        "--n-splits-cv", "3", "--n-jobs", "1", "--n-iter", "3",
                        "--lr-penalty", self.args.lr_penalty, "--xgb-policy", self.args.xgb_policy, *self.common())
        if self.args.skip_explanations:
            argv += ["--skip-shap", "--skip-svm-importance"]
        elif self.args.skip_svm_importance:
            argv += ["--skip-svm-importance"]
        self.command(argv, output / "command.log")
        check_evaluation(output / "06_train_eval", frame(cached["train"]))
        check_evaluation(output / "07_test_eval", frame(cached["holdout"]))
        artifact = one(output / "04_model", "*.joblib")
        loaded = joblib.load(artifact)
        fitted = loaded["model"] if isinstance(loaded, dict) else loaded
        encoder = loaded.get("label_encoder") if isinstance(loaded, dict) else None
        if model == "XGBC" or sampling in {"smote", "enn", "smoteenn"}:
            require(encoder is not None, "Encoded training artifact omitted label encoder")
        ext = self.args.fixtures / "dataset3_test" / dataset
        truth = frame(ext / "metadata.tsv")
        feature_names = frame(cached["train"]).drop(columns=["source_summary", "cluster5"]).columns.tolist()
        stored_names = fitted.mgl_feature_names if hasattr(fitted, "mgl_feature_names") else fitted.feature_names_in_
        require(list(stored_names) == feature_names, "Artifact feature order")
        if encoder is not None:
            require(list(encoder.classes_) == sorted(truth.source_summary.unique()), "Encoder classes changed")
        else:
            require(list(fitted.classes_) == sorted(truth.source_summary.unique()), "Native string classes changed")
        raw = frame(ext / "predict_features.tsv")[feature_names]
        direct = fitted.predict(raw)
        if encoder is not None:
            direct = encoder.inverse_transform(direct.astype(int))
        reloaded = joblib.load(artifact)
        second_model = reloaded["model"] if isinstance(reloaded, dict) else reloaded
        second_encoder = reloaded.get("label_encoder") if isinstance(reloaded, dict) else None
        second_predictions = second_model.predict(raw)
        if second_encoder is not None:
            second_predictions = second_encoder.inverse_transform(second_predictions.astype(int))
        np.testing.assert_array_equal(direct, second_predictions)
        np.testing.assert_allclose(fitted.predict_proba(raw), second_model.predict_proba(raw), atol=0, rtol=0)
        inputs = output / "external_inputs"
        inputs.mkdir()
        labelled = pd.concat([truth[["source_summary", "cluster5"]], raw], axis=1)
        labelled.to_csv(inputs / "ready.tsv", sep="\t")
        raw.to_csv(inputs / "predict.tsv", sep="\t")
        route_predictions = {}
        for route in ["labelled_ready", "labelled_extraction", "predict_ready", "predict_extraction"]:
            destination = output / route
            argv = self.cli("test", "--model-file", artifact, "--name", route, "--output-dir", destination, *self.common())
            if route.endswith("ready"):
                argv += ["--features-test", str(inputs / ("predict.tsv" if route.startswith("predict") else "ready.tsv"))]
            else:
                argv += ["--features", str(ext / "features.tsv"), "--feature-file", str(cached["feature_file"])]
                if route.startswith("labelled"):
                    argv += ["--test-metadata", str(ext / "metadata.tsv")]
            if route.startswith("predict"):
                argv += ["--predict-only"]
            self.command(argv, destination / "command.log")
            evaluation = destination / "07_test_eval"
            if route.startswith("labelled"):
                predicted = check_evaluation(evaluation, labelled)
                route_predictions[route] = predicted
                np.testing.assert_array_equal(predicted.loc[raw.index, "prediction"], direct)
            else:
                # Existing evaluator may call its reset index 'index' after concat.
                # Check actual IDs, independent of that first-column header.
                predicted = frame(one(evaluation, "*_predictions.tsv"))
                require(predicted.index.is_unique and set(predicted.index) == set(raw.index), "Unlabelled sample coverage")
                np.testing.assert_array_equal(predicted.loc[raw.index, "Prediction"], direct)
                route_predictions[route] = predicted.rename(columns={"Prediction": "prediction"})
        classes = sorted(truth.source_summary.unique())
        baseline = route_predictions["labelled_ready"].loc[raw.index]
        check_artifact_probabilities(artifact, raw, baseline, classes)
        for route, predicted in route_predictions.items():
            require([c for c in predicted.columns if c in classes] == classes, "Prediction class order")
            np.testing.assert_allclose(predicted.loc[raw.index, classes], baseline[classes], atol=1e-10)
        explanations = {"skipped_by_option": self.args.skip_explanations,
                        "svm_importance_skipped": model == "SVM" and (self.args.skip_explanations or self.args.skip_svm_importance),
                        "importance_files": [str(p) for p in (output / "06_train_eval").glob("*_feature_importances.tsv")],
                        "shap_files": [str(p) for p in (output / "06_train_eval").glob("*_shap_values.tsv")]}
        explanations["shap_status"] = ("skipped_by_option" if self.args.skip_explanations else
                                       "not_applicable" if model in {"LR", "SVM"} else
                                       "written" if explanations["shap_files"] else "failed_or_skipped_see_log")
        if not (model == "SVM" and (self.args.skip_svm_importance or self.args.skip_explanations)):
            importance = frame(one(output / "06_train_eval", "*_feature_importances.tsv"))
            require(importance.index.is_unique and set(importance.index) == set(feature_names), "Explanation feature coverage")
            require(np.isfinite(importance.to_numpy()).all(), "Nonfinite feature importance")
        if not self.args.skip_explanations:
            if not (model == "SVM" and self.args.skip_svm_importance):
                importance = frame(one(output / "06_train_eval", "*_feature_importances.tsv"))
                require(set(importance.index) == set(feature_names), "Explanation feature coverage")
                require(np.isfinite(importance.to_numpy()).all(), "Nonfinite feature importance")
            for path in (output / "06_train_eval").glob("*_shap_values.tsv"):
                values = frame(path)
                require(set(values.index) == set(frame(cached["train"]).index), "SHAP sample coverage")
                require(list(values.columns) == feature_names, "SHAP feature order")
                require(np.isfinite(values.to_numpy()).all(), "Nonfinite SHAP values")
            if model in {"RFC", "XGBC"}:
                require(len(explanations["shap_files"]) == len(classes), "SHAP class coverage")
                require(all(any(Path(p).name.endswith("_" + label + "_shap_values.tsv")
                                for p in explanations["shap_files"]) for label in classes), "SHAP class names")
        return {"artifact": str(artifact), "features": len(feature_names), "external_routes": list(route_predictions),
                "explanations": explanations}

    def execute_case(self, dataset, directory, selection, cached, model, sampling):
        started = time.monotonic()
        result = {"kind": "model", "dataset": dataset, "selection": selection,
                  "model": model, "sampling": sampling}
        print("Running", dataset, selection, sampling, model, flush=True)
        try:
            result.update(self.case(dataset, directory, selection, cached, model, sampling))
            result["status"] = "passed"
        except Exception as error:
            result.update(status="failed", error=str(error))
            if eligible_enn_rejection(result):
                case = self.output / dataset / "models" / ("%s_%s_%s_%s" % (dataset, selection, model, sampling))
                try:
                    evidence = verify_enn_rejection(result, cached["train"], case, self.report["commands"][-1])
                    path = case / "expected_rejection.json"
                    path.write_text(json.dumps(evidence, indent=2) + "\n")
                    result.update(status="expected_rejection", rejection=evidence, evidence_file=str(path),
                                  evidence_sha256=checksum(path),
                                  reason="Independently verified exhausted ENN fallback; no model was trained")
                except Exception as verification_error:
                    result["rejection_verification_error"] = str(verification_error)
        result["seconds"] = time.monotonic() - started
        self.results.append(result)
        self.save()
        print(result["status"], result.get("reason", result.get("error", "")), flush=True)
        return result

    def run(self):
        for dataset in self.args.datasets:
            directory = self.args.fixtures / ("dataset1_multiclass" if dataset == "multiclass" else "dataset2_binary")
            self.preflight(dataset, directory)
            chisq = None
            for selection in self.args.selections:
                cache = {"train": directory / "ready_train.tsv", "holdout": directory / "ready_holdout.tsv",
                         "feature_file": directory / "all_features.tsv"}
                if selection != "none":
                    started = time.monotonic()
                    try:
                        cache = self.selector(dataset, directory, selection, chisq)
                        chisq = cache["chisq"]
                        self.results.append({"kind": "selector", "dataset": dataset, "selection": selection,
                                             "status": "passed", "exports": cache["exports"], "checksums": cache["checksums"],
                                             "cache_reused": False, "seconds": time.monotonic() - started})
                    except Exception as error:
                        self.results.append({"kind": "selector", "dataset": dataset, "selection": selection,
                                             "status": "failed", "error": str(error), "seconds": time.monotonic() - started})
                        for model in self.args.models:
                            for sampling in self.args.samplings:
                                self.results.append({"kind": "model", "dataset": dataset, "selection": selection,
                                                     "model": model, "sampling": sampling, "status": "blocked_by_selector"})
                        self.save()
                        continue
                for sampling in self.args.samplings:
                    for model in self.args.models:
                        self.execute_case(dataset, directory, selection, cache, model, sampling)
        self.save()
        self.report["summary"] = dict(collections.Counter(r["status"] for r in self.results if r["kind"] == "model"))
        export_model_index(self.report, self.output / "model_index.tsv")
        self.save()
        rejections = [r for r in self.results if r["status"] == "expected_rejection"]
        return int(len(rejections) > 1 or any(
            r["status"] != "passed" and not (r["status"] == "expected_rejection" and not expected_rejection_errors(r))
            for r in self.results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "test")
    parser.add_argument("--output", required=True, type=Path, help="New output directory; existing directories are rejected")
    parser.add_argument("--datasets", nargs="+", choices=["multiclass", "binary"], default=["multiclass", "binary"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--selections", nargs="+", choices=SELECTIONS, default=SELECTIONS)
    parser.add_argument("--samplings", nargs="+", choices=SAMPLINGS, default=["none"])
    parser.add_argument("--lr-penalty", choices=["l1", "l2", "elasticnet"], default="l2")
    parser.add_argument("--xgb-policy", choices=["depthwise", "lossguide"], default="depthwise")
    parser.add_argument("--skip-explanations", action="store_true", help="Explicit partial coverage for faster diagnostic runs")
    parser.add_argument("--skip-svm-importance", action="store_true", help="Explicitly omit expensive SVM permutation importance")
    parser.add_argument("--input-format", choices=["tsv", "csr"], default="tsv")
    args = parser.parse_args()
    args.fixtures = args.fixtures.resolve()
    if args.input_format == "csr":
        from sparse import SparseRunner
        runner = SparseRunner(args)
    else:
        runner = Runner(args)
    sys.exit(runner.run())


if __name__ == "__main__":
    main()

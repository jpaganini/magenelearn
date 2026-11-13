#!/usr/bin/env python3
"""Run portable quick or full maGeneLearn validation from a repository checkout."""
import argparse
import collections
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET

HELPERS = Path(__file__).resolve().parent / "validation"
sys.path.insert(0, str(HELPERS))
from common import ROOT, checksum, run_process
from matrix import expected_rejection_errors

MODELS = ["RFC", "XGBC", "SVM", "LR"]
SELECTIONS = ["none", "muvr_RFC", "muvr_XGBC", "boruta_RFC", "boruta_XGBC"]
SAMPLINGS = ["none", "random", "smote", "random_under", "enn", "smoteenn"]


def approved_rejection_count(results, profile):
    claims = [r for r in results if r.get("status") == "expected_rejection"]
    if len(claims) > 1:
        raise ValueError("At most one expected ENN rejection is permitted")
    for result in claims:
        errors = expected_rejection_errors(result)
        if profile != "full" or result.get("stage") != "matrix" or errors:
            raise ValueError("Unapproved expected rejection: " + "; ".join(errors or ["Wrong profile or stage"]))
    return len(claims)


def expected_artifact_audits(results, profile):
    # Never derive the allowance from the number of files/CSV rows found.
    return (246 if profile == "full" else 40) - approved_rejection_count(results, profile)


def coverage_errors(results, profile):
    expected = set(itertools.product(["multiclass", "binary"], SELECTIONS,
                                     SAMPLINGS if profile == "full" else ["none"], MODELS))
    keys = [(r.get("dataset"), r.get("selection"), r.get("sampling"), r.get("model"))
            for r in results if r.get("kind") == "model"]
    errors = []
    if set(keys) != expected or len(keys) != len(expected):
        errors.append("Matrix coverage mismatch: expected %d unique cases, received %d; missing=%s; unexpected=%s" %
                      (len(expected), len(keys), sorted(expected - set(keys)), sorted(set(keys) - expected)))
    try:
        approved_rejection_count([{**r, "stage": "matrix"} for r in results], profile)
    except ValueError as error:
        errors.append(str(error))
    return errors


def write_reports(report, output):
    """Write reports even for setup errors and interrupted/blocked stages."""
    results = report["results"]
    counts = collections.Counter(r["status"] for r in results)
    try:
        approved_rejection_count(results, report["profile"])
        rejections_valid = True
    except ValueError:
        rejections_valid = False

    def accepted(result):
        return result["status"] in {"passed", "omitted"} or (rejections_valid and result["status"] == "expected_rejection")

    failed = any(not accepted(r) for r in results)
    report["summary"] = dict(counts)
    report["status"] = "failed" if failed else "passed" if report.get("finished_utc") else "running"
    report["validation_scope"] = report["profile"] + " profile only"
    matrix = [r for r in results if r.get("stage") == "matrix" and r.get("kind") == "model"]
    def coverage(cases, expected):
        trained = sum(r["status"] == "passed" for r in cases)
        rejected = sum(r["status"] == "expected_rejection" and accepted(r) for r in cases)
        return {"passed": trained, "trained": trained, "expected_rejections": rejected,
                "accepted": trained + rejected, "unexpected_failures": sum(not accepted(r) for r in cases),
                "recorded": len(cases), "expected": expected}

    report["coverage"] = {"matrix": coverage(matrix, 240 if report["profile"] == "full" else 40),
                          "matrix_by_problem": {problem: coverage([r for r in matrix if r.get("dataset") == problem],
                                                                  120 if report["profile"] == "full" else 20)
                                                for problem in ["binary", "multiclass"]}}
    csr = [r for r in results if r.get("stage") == "csr_matrix" and r.get("kind") == "model"]
    report["coverage"]["csr_matrix"] = coverage(csr, 240 if report["profile"] == "full" else 40)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    suite = ET.Element("testsuite", name="maGeneLearn-" + report["profile"], tests=str(len(results)),
                       failures=str(sum(not accepted(r) for r in results)),
                       skipped=str(counts["omitted"]))
    for result in results:
        case = ET.SubElement(suite, "testcase", name=result["name"], time=str(result.get("seconds", 0)))
        if result["status"] == "omitted":
            ET.SubElement(case, "skipped", message=result.get("reason", "Explicit profile omission"))
        elif result["status"] == "expected_rejection" and accepted(result):
            props = ET.SubElement(case, "properties")
            ET.SubElement(props, "property", name="outcome", value="expected_rejection")
            ET.SubElement(props, "property", name="evidence", value=result["evidence_file"])
        elif not accepted(result):
            ET.SubElement(case, "failure", message=result.get("error", result["status"])).text = result.get("error", result["status"])
        if result.get("log"):
            ET.SubElement(case, "system-out").text = result["log"] + (
                "\n" + result.get("reason", "") + "\nEvidence: " + result.get("evidence_file", "")
                if result["status"] == "expected_rejection" else "")
    ET.ElementTree(suite).write(output / "junit.xml", encoding="unicode", xml_declaration=True)
    lines = ["# maGeneLearn validation: " + report["status"], "", "Profile: **" + report["profile"] + "**",
             "", "This result covers the declared profile, not every possible input or configuration.", "",
             "| Status | Count |", "|---|---:|"]
    lines += ["| %s | %d |" % pair for pair in sorted(counts.items())]
    matrix_counts = report["coverage"]["matrix"]
    lines += ["", "Main model matrix: **%d trained; %d verified expected rejections; %d unexpected failures** (%d required configurations)." %
              (matrix_counts["trained"], matrix_counts["expected_rejections"], matrix_counts["unexpected_failures"], matrix_counts["expected"])]
    csr_counts = report["coverage"]["csr_matrix"]
    lines += ["", "CSR model matrix: **%d trained; %d unexpected failures** (%d required configurations)." %
              (csr_counts["trained"], csr_counts["unexpected_failures"], csr_counts["expected"])]
    lines += ["", "## Expected rejections, failures and omissions", ""]
    for r in results:
        if r["status"] != "passed":
            lines.append("- **%s**: %s — %s" % (r["status"], r["name"], r.get("reason", r.get("error", ""))))
            if r.get("log"):
                lines.append("  Log: `%s`" % r["log"])
            if r.get("evidence_file"):
                lines.append("  Verification evidence: `%s`" % r["evidence_file"])
    lines += ["", "See `report.json` and per-stage reports for commands, exact coverage, model inputs, and provenance."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return int(failed)


class Validation:
    def __init__(self, profile, output):
        self.profile, self.output = profile, output.resolve()
        self.started = time.monotonic()
        self.output.mkdir(parents=True, exist_ok=False)
        self.env = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0",
                    "MPLCONFIGDIR": str(self.output / "matplotlib_cache")}
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]:
            self.env[key] = "1"
        self.report = {"schema_version": 2, "profile": profile, "results": [], "stages": [],
                       "python": sys.version, "executable": sys.executable,
                       "started_utc": datetime.now(timezone.utc).isoformat(),
                       "seeds": {"split_boruta": 42, "training_evaluation": 50, "muvr": None},
                       "concurrency": {k: self.env[k] for k in self.env if k.endswith("NUM_THREADS") or k == "LOKY_MAX_CPU_COUNT"},
                       "dense_workspace_limit_mb": self.env.get("MAGENELEARN_MAX_DENSE_MB", "1024"),
                       "limitations": ["Functionality fixtures, not independent performance cohorts",
                                       "Feature selection precedes downstream CV; tuned evaluation is not nested CV",
                                       "MUVR has no explicit seed or group vector",
                                       "SVM full-feature importance uses 36/24 training-only diagnostic samples",
                                       "Reference environment is Linux/Python 3.9; count-valued inputs and all legacy/scoring variants are not exhaustive"]}

    def save(self):
        return write_reports(self.report, self.output)

    def stage(self, name, argv, report_path=None, expected=None, timeout=14400):
        log = self.output / (name + ".log")
        start = time.monotonic()
        entry = {"name": name, "command": list(map(str, argv)), "log": str(log), "status": "running"}
        self.report["stages"].append(entry)
        self.save()
        print("Running " + name, flush=True)
        try:
            with log.open("x") as handle:
                result = run_process(entry["command"], cwd=ROOT, env=self.env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=timeout)
            entry["exit_code"] = result.returncode
            if name == "unit_and_fixture_tests":
                count = re.search(r"Ran (\d+) tests? in", log.read_text())
                if count:
                    self.report["unit_tests_executed"] = int(count.group(1))
            if report_path is not None:
                child = json.loads(report_path.read_text())
                cases = child["results"]
                if expected is not None and len(cases) != expected:
                    raise ValueError("Expected %d results, found %d" % (expected, len(cases)))
                for index, case in enumerate(cases):
                    label = case.get("check") or "/".join(str(case.get(k, "")) for k in ["kind", "dataset", "selection", "model", "sampling"])
                    self.report["results"].append({**case, "stage": name, "name": name + "/" + label + "/" + str(index), "log": str(log)})
                if name in {"matrix", "csr_matrix"}:
                    errors = coverage_errors(cases, self.profile)
                    if errors:
                        raise ValueError("; ".join(errors))
            if result.returncode:
                raise ValueError("Command exited %d; see %s" % (result.returncode, log))
            if report_path is not None:
                approved_rejection_count([{**r, "stage": name} for r in cases], self.profile)
                if any(r["status"] not in {"passed", "expected_rejection"} for r in cases):
                    raise ValueError("Stage reported required failed or blocked cases")
            entry["status"] = "passed"
        except Exception as error:
            entry.update(status="failed", error=str(error))
        entry["seconds"] = time.monotonic() - start
        self.report["results"].append({k: v for k, v in entry.items() if k != "command"})
        self.save()
        print(name + ": " + entry["status"], flush=True)
        return entry["status"] == "passed"

    def provenance(self):
        manifest = ROOT / "test/manifest.json"
        self.report["fixture_manifest_sha256"] = checksum(manifest)
        self.report["fixture_files_sha256"] = {str(p.relative_to(ROOT)): checksum(p) for p in sorted((ROOT / "test").rglob("*")) if p.is_file()}
        paths = list((ROOT / "maGeneLearn").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))
        self.report["source_sha256"] = {str(p.relative_to(ROOT)): checksum(p) for p in sorted(paths)}
        def git(*args):
            result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
            return result.stdout.strip() if result.returncode == 0 else "unavailable"
        self.report["git"] = {"revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
                              "status": git("status", "--porcelain")}
        import ast
        setup = ast.parse((ROOT / "setup.py").read_text())
        pins = next(ast.literal_eval(k.value) for node in ast.walk(setup) if isinstance(node, ast.Call)
                    for k in node.keywords if k.arg == "install_requires")
        self.report["versions"] = {}
        for requirement in pins:
            name, version = requirement.split("==")
            actual = importlib.metadata.version(name)
            self.report["versions"][name] = actual
            if actual != version:
                raise ValueError("Install repository dependencies: %s requires %s, found %s" % (name, version, actual))
        for package in ["scipy", "setuptools", "pip"]:
            self.report["versions"][package] = importlib.metadata.version(package)
        self.report["installed_packages"] = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}

    def run(self):
        required = {"unit_and_fixture_tests", "help_root", "help_train", "help_test", "matrix", "targeted", "data_edges", "full_train", "artifact_audit"}
        required.update(["csr_matrix", "csr_targeted", "csr_data_edges", "csr_full_train", "csr_artifact_audit"])
        if self.profile == "full":
            required.update(["lr_l1", "lr_elasticnet", "xgb_lossguide", "svm_importance"])
            required.update(["csr_lr_l1", "csr_lr_elasticnet", "csr_xgb_lossguide", "csr_svm_importance"])
        self.report["required_stages"] = sorted(required)
        try:
            self.provenance()
            self.save()
            if not self.stage("unit_and_fixture_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"], timeout=600):
                raise ValueError("Unit/fixture tests failed; integration stages blocked")
            for command in [[], ["train"], ["test"]]:
                self.stage("help_" + (command[0] if command else "root"),
                           [sys.executable, "-c", "from maGeneLearn.cli import cli; cli()", *command, "--help"], timeout=60)
            matrix = self.output / "matrix"
            argv = [sys.executable, HELPERS / "matrix.py", "--output", matrix, "--skip-svm-importance"]
            argv += ["--samplings", *SAMPLINGS] if self.profile == "full" else ["--skip-explanations"]
            # Eight selector records plus the declared model matrix.
            self.stage("matrix", argv, matrix / "report.json", 248 if self.profile == "full" else 48)
            indices = [matrix / "model_index.tsv"]
            runtime = [sys.executable, HELPERS / "runtime.py", "--models-run", matrix]
            for name, flag, expected in [("targeted", None, 32), ("data_edges", "--only-data-edges", 6),
                                         ("full_train", "--only-orchestration", 4)]:
                directory = self.output / name
                args = runtime + ["--output", directory] + ([flag] if flag else [])
                self.stage(name, args, directory / "report.json", expected)
            if self.profile == "full":
                for name, option, value, model in [("lr_l1", "--lr-penalty", "l1", "LR"),
                                                   ("lr_elasticnet", "--lr-penalty", "elasticnet", "LR"),
                                                   ("xgb_lossguide", "--xgb-policy", "lossguide", "XGBC")]:
                    directory = self.output / name
                    self.stage(name, [sys.executable, HELPERS / "matrix.py", "--output", directory,
                                      "--selections", "none", "--models", model, option, value, "--skip-svm-importance"],
                               directory / "report.json", 2)
                    indices.append(directory / "model_index.tsv")
                directory = self.output / "svm_importance"
                self.stage("svm_importance", runtime + ["--output", directory, "--only-svm-importance"],
                           directory / "report.json", 2, timeout=18000)
            else:
                for name in ["tree_shap", "svm_permutation_importance", "nondefault_saved_model_variants", "five_additional_sampling_settings"]:
                    self.report["results"].append({"name": name, "status": "omitted", "reason": "Not required by quick profile; covered by full profile"})
            existing = [p for p in indices if p.is_file()]
            audit = self.output / "artifact_audit"
            self.stage("artifact_audit", [sys.executable, HELPERS / "audit.py", "--profile", self.profile,
                                          "--model-index", *existing, "--output", audit], audit / "report.json",
                       expected_artifact_audits(self.report["results"], self.profile))
            csr_matrix = self.output / "csr_matrix"
            argv = [sys.executable, HELPERS / "matrix.py", "--input-format", "csr", "--output", csr_matrix, "--skip-svm-importance"]
            argv += ["--samplings", *SAMPLINGS] if self.profile == "full" else ["--skip-explanations"]
            self.stage("csr_matrix", argv, csr_matrix / "report.json", 248 if self.profile == "full" else 48)
            csr_indices = [csr_matrix / "model_index.tsv"]
            csr_runtime = [sys.executable, HELPERS / "runtime.py", "--input-format", "csr", "--models-run", csr_matrix]
            for name, flag, expected in [("csr_targeted", None, 32), ("csr_data_edges", "--only-data-edges", 6),
                                         ("csr_full_train", "--only-orchestration", 4)]:
                directory = self.output / name
                self.stage(name, csr_runtime + ["--output", directory] + ([flag] if flag else []),
                           directory / "report.json", expected)
            if self.profile == "full":
                for name, option, value, model in [("csr_lr_l1", "--lr-penalty", "l1", "LR"),
                                                   ("csr_lr_elasticnet", "--lr-penalty", "elasticnet", "LR"),
                                                   ("csr_xgb_lossguide", "--xgb-policy", "lossguide", "XGBC")]:
                    directory = self.output / name
                    self.stage(name, [sys.executable, HELPERS / "matrix.py", "--input-format", "csr", "--output", directory,
                                      "--selections", "none", "--models", model, option, value, "--skip-svm-importance"],
                               directory / "report.json", 2)
                    csr_indices.append(directory / "model_index.tsv")
                directory = self.output / "csr_svm_importance"
                self.stage("csr_svm_importance", csr_runtime + ["--output", directory, "--only-svm-importance"],
                           directory / "report.json", 2, timeout=18000)
            csr_existing = [p for p in csr_indices if p.is_file()]
            audit = self.output / "csr_artifact_audit"
            self.stage("csr_artifact_audit", [sys.executable, HELPERS / "audit.py", "--profile", self.profile,
                                              "--model-index", *csr_existing, "--output", audit], audit / "report.json",
                       246 if self.profile == "full" else 40)
            existing += csr_existing
            self.combine_indices(existing)
            for rel, original in self.report["fixture_files_sha256"].items():
                if checksum(ROOT / rel) != original:
                    raise ValueError("Validation modified a reference fixture: " + rel)
        except (Exception, KeyboardInterrupt) as error:
            self.report["results"].append({"name": "validation_setup_or_interruption", "status": "failed",
                                           "error": str(error) or type(error).__name__, "traceback": traceback.format_exc()})
        finally:
            completed = {r["name"] for r in self.report["results"]}
            for missing in sorted(required - completed):
                self.report["results"].append({"name": missing, "status": "blocked", "error": "Required stage did not complete"})
            self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self.report["wall_seconds"] = time.monotonic() - self.started
            self.stage_failed_artifacts()
            code = self.save()
            print("Report: " + str(self.output / "summary.md"), flush=True)
        return code

    def combine_indices(self, indices):
        import csv
        rows, fields = [], None
        for index in indices:
            with index.open() as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if fields is not None and fields != reader.fieldnames:
                    raise ValueError("Incompatible model-index schemas")
                fields = reader.fieldnames
                rows.extend(reader)
        if not fields:
            return
        with (self.output / "models_for_dataset3.tsv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def stage_failed_artifacts(self):
        # Keep expected rejections separate from unexpected failure evidence.
        for stage in self.report["stages"]:
            path = self.output / stage["name"] / "report.json"
            if not path.is_file():
                continue
            try:
                results = json.loads(path.read_text())["results"]
                for case in results:
                    if case.get("status") == "passed":
                        continue
                    if case.get("kind") == "model":
                        name = "_".join(str(case[k]) for k in ["dataset", "selection", "model", "sampling"])
                        directory = path.parent / case["dataset"] / "models" / name
                    elif case.get("artifact"):
                        directory = Path(case["artifact"]).parents[1]
                        name = directory.name
                    elif stage["name"] in {"full_train", "csr_full_train"}:
                        dataset, _, route = case["check"].split("/")
                        name = dataset + "_" + route
                        directory = path.parent / name
                    elif stage["name"] in {"svm_importance", "csr_svm_importance"}:
                        name = case["check"].split("/")[-1]
                        directory = path.parent / name
                    else:
                        continue
                    if directory.is_dir():
                        expected = (case.get("status") == "expected_rejection" and self.profile == "full" and
                                    stage["name"] == "matrix" and not expected_rejection_errors(case))
                        folder = "expected_rejections" if expected else "failed_cases"
                        shutil.copytree(directory, self.output / folder / stage["name"] / name)
            except Exception as error:
                self.report["results"].append({"name": "failure_evidence/" + stage["name"], "status": "failed", "error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "full"], default="full")
    parser.add_argument("--output", type=Path, help="New result directory (default: unique validation_results directory)")
    args = parser.parse_args()
    output = args.output or ROOT / "validation_results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    if output.exists():
        parser.error("Output directory already exists; refusing to overwrite: " + str(output))
    return Validation(args.profile, output).run()


if __name__ == "__main__":
    sys.exit(main())

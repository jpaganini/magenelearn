"""Test the validator's ability to reject missing evidence and wrong probabilities."""
import itertools
import json
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/validation"))
from common import run_process
from matrix import (check_artifact_probabilities, check_evaluation, frame, checksum,
                    verify_enn_rejection, expected_rejection_errors, export_model_index, Runner)
from validate import (coverage_errors, write_reports, MODELS, SELECTIONS, SAMPLINGS,
                      Validation, expected_artifact_audits)


class ValidationRegression(unittest.TestCase):
    def test_missing_artifact_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Expected one"):
                check_evaluation(Path(tmp), None)

    def test_altered_probability_is_detected_even_when_sum_is_one(self):
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier
        raw = pd.DataFrame({"feature": [0, 1, 0, 1]}, index=["a", "b", "c", "d"])
        model = DummyClassifier(strategy="prior").fit(raw, ["Bovine", "Bovine", "Bovine", "Swine"])
        probabilities = pd.DataFrame(model.predict_proba(raw), index=raw.index, columns=model.classes_)
        probabilities["prediction"] = model.predict(raw)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "model.joblib"
            joblib.dump(model, artifact)
            check_artifact_probabilities(artifact, raw, probabilities, ["Bovine", "Swine"])
            probabilities.loc["a", ["Bovine", "Swine"]] = [0.6, 0.4]
            with self.assertRaises(AssertionError):
                check_artifact_probabilities(artifact, raw, probabilities, ["Bovine", "Swine"])

    def test_round_trip_reader_preserves_ranking(self):
        from sklearn.metrics import average_precision_score
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.tsv"
            path.write_text("Sample.ID\tscore\na\t0.1355125536124339\nb\t0.13551255361243392\n")
            scores = frame(path).score
            self.assertEqual(scores.nunique(), 2)
            self.assertEqual(average_precision_score([0, 1], scores), 1)

    def test_missing_or_duplicate_matrix_case_is_detected(self):
        cases = [{"kind": "model", "dataset": d, "selection": s, "sampling": "none", "model": m}
                 for d, s, m in itertools.product(["multiclass", "binary"], SELECTIONS, MODELS)]
        self.assertEqual(coverage_errors(cases, "quick"), [])
        self.assertTrue(coverage_errors(cases[:-1], "quick"))
        self.assertTrue(coverage_errors(cases + cases[:1], "quick"))
        self.assertTrue(coverage_errors(cases, "full"))

    def test_saved_model_and_sampler_configuration_is_checked(self):
        from types import SimpleNamespace
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.over_sampling import RandomOverSampler
        from audit import check_configuration
        fitted = SimpleNamespace(named_steps={"model": RandomForestClassifier(), "sampler": RandomOverSampler()})
        check_configuration(fitted, "RFC", "random")
        with self.assertRaisesRegex(ValueError, "sampler differs"):
            check_configuration(fitted, "RFC", "none")
        with self.assertRaisesRegex(ValueError, "classifier differs"):
            check_configuration(fitted, "LR", "random")

    def test_missing_or_incorrect_tuning_output_is_detected(self):
        from audit import check_tuning
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Expected one"):
                check_tuning(directory)
            path = directory / "trials.tsv"
            header = "trial\tscore_mean\tscore_std\tfold0_score\tfold1_score\tfold2_score\n"
            path.write_text(header + "0\t0.5\t0\t0.5\t0.5\t0.5\n")
            check_tuning(directory)
            path.write_text(header + "0\t0.6\t0\t0.5\t0.5\t0.5\n")
            with self.assertRaises(AssertionError):
                check_tuning(directory)

    def test_report_failure_omission_and_running_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            report = {"profile": "quick", "results": []}
            self.assertEqual(write_reports(report, path), 0)
            self.assertEqual(report["status"], "running")
            report["finished_utc"] = "test"
            report["results"] = [{"name": "shap", "status": "omitted", "reason": "quick profile"}]
            self.assertEqual(write_reports(report, path), 0)
            self.assertEqual(report["validation_scope"], "quick profile only")
            report["results"].append({"name": "artifact", "status": "blocked", "error": "Missing model"})
            self.assertEqual(write_reports(report, path), 1)
            self.assertEqual(json.loads((path / "report.json").read_text())["status"], "failed")
            suite = ET.parse(path / "junit.xml").getroot()
            self.assertEqual(suite.attrib["failures"], "1")
            self.assertEqual(suite.attrib["skipped"], "1")

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            sentinel = path / "preserved.txt"
            sentinel.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                Validation("quick", path)
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_subprocess_capture_and_timeout(self):
        result = run_process([sys.executable, "-c", "print('captured')"], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "captured")
        with self.assertRaises(subprocess.TimeoutExpired):
            run_process([sys.executable, "-c", "import time; time.sleep(20)"], timeout=0.1)

    def test_timeout_terminates_descendant_process_groups(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child_pid"
            program = ("import subprocess,sys,time; from pathlib import Path; "
                       "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
                       "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)")
            with self.assertRaises(subprocess.TimeoutExpired):
                run_process([sys.executable, "-c", program, str(marker)], timeout=1)
            pid = int(marker.read_text())
            self.assertTrue(not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE)


class ExpectedENNRejectionRegression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / "matrix/multiclass/models/multiclass_muvr_RFC_XGBC_enn"
        (self.directory / "04_model").mkdir(parents=True)
        self.train = self.root / "train.tsv"
        original = frame(ROOT / "test/dataset1_multiclass/features.tsv")
        metadata = frame(ROOT / "test/dataset1_multiclass/train_metadata.tsv")
        data = original.loc[original.index.isin(metadata.index), ["STMMW_22311__1", "STMMW_43051__22"]].copy()
        data["source_summary"] = metadata.loc[data.index, "source_summary"]
        data["cluster5"] = metadata.loc[data.index, "cluster5"]
        data.to_csv(self.train, sep="\t")
        self.command_log = self.directory / "command.log"
        self.command_log.write_text("Step 'train' failed with exit code 1.\n")
        self.training_log = self.directory / "04_model/train.log"
        self.training_log.write_text(
            "Traceback (most recent call last):\n"
            "sampling_validation.SamplingClassLoss: Optimization trial 0: "
            "no ENN neighbour count from 5 to 1 preserved every class. Sampling was not disabled.\n")
        self.command = {"exit_code": 1, "log": str(self.command_log),
                        "command": [sys.executable, "-c", "training test double", "train",
                                    "--features-train", str(self.train)]}
        self.result = {"kind": "model", "dataset": "multiclass", "selection": "muvr_RFC",
                       "model": "XGBC", "sampling": "enn", "status": "failed"}

    def verified(self):
        evidence = verify_enn_rejection(self.result, self.train, self.directory, self.command)
        path = self.directory / "expected_rejection.json"
        path.write_text(json.dumps(evidence))
        return {**self.result, "status": "expected_rejection", "stage": "matrix", "name": "approved_enn",
                "reason": "Independently verified exhausted ENN", "rejection": evidence,
                "evidence_file": str(path), "evidence_sha256": checksum(path), "log": str(self.command_log)}

    def test_real_enn_confirms_all_candidates_and_partitions(self):
        result = self.verified()
        self.assertEqual(expected_rejection_errors(result), [])
        self.assertEqual([c["neighbors"] for c in result["rejection"]["candidates"]], [5, 4, 3, 2, 1])
        self.assertTrue(all(len(c["counts"]) == 4 for c in result["rejection"]["candidates"]))
        self.assertEqual(result["rejection"]["candidates"][0]["counts"][0]["after"]["Swine"], 0)

    def test_other_configuration_is_not_eligible(self):
        for key, value in [("dataset", "binary"), ("selection", "muvr_XGBC"),
                           ("model", "RFC"), ("sampling", "smoteenn")]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "outside the approved case"):
                verify_enn_rejection({**self.result, key: value}, self.train, self.directory, self.command)

    def test_wrong_failure_phase_timeout_and_exit_code_are_rejected(self):
        for change in [{"status": "timed_out"}, {"exit_code": 2},
                       {"log": str(self.directory / "07_test_eval/command.log")}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                verify_enn_rejection(self.result, self.train, self.directory, {**self.command, **change})
        self.command_log.write_text("Step 'eval_train' failed with exit code 1.\n")
        with self.assertRaisesRegex(ValueError, "not in the training step"):
            verify_enn_rejection(self.result, self.train, self.directory, self.command)

    def test_unrelated_terminal_error_is_not_accepted(self):
        self.training_log.write_text(self.training_log.read_text() + "RuntimeError: unrelated failure\n")
        with self.assertRaisesRegex(ValueError, "terminal training error"):
            verify_enn_rejection(self.result, self.train, self.directory, self.command)
        self.training_log.write_text("FileNotFoundError: missing input\n")
        with self.assertRaisesRegex(ValueError, "No unambiguous"):
            verify_enn_rejection(self.result, self.train, self.directory, self.command)

    def test_model_and_evaluation_artifacts_prevent_rejection(self):
        for rel in ["04_model/model.joblib", "07_test_eval/metrics.tsv", "predict_ready/predictions.tsv"]:
            path = self.directory / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unexpected artifact")
            with self.subTest(path=rel), self.assertRaisesRegex(ValueError, "produced model/evaluation"):
                verify_enn_rejection(self.result, self.train, self.directory, self.command)
            path.unlink()

    def test_candidate_that_preserves_classes_prevents_rejection(self):
        from unittest.mock import patch
        # The verifier must catch a false production claim even if just one
        # candidate preserves classes on all four partitions.
        from imblearn.under_sampling import EditedNearestNeighbours
        original = EditedNearestNeighbours.fit_resample

        def sometimes_valid(sampler, X, y):
            return (X, y) if sampler.n_neighbors == 3 else original(sampler, X, y)

        with patch.object(EditedNearestNeighbours, "fit_resample", sometimes_valid):
            with self.assertRaisesRegex(ValueError, "candidate preserves every class"):
                verify_enn_rejection(self.result, self.train, self.directory, self.command)

    def test_missing_or_changed_evidence_is_not_accepted(self):
        valid = self.verified()
        for key in ["candidates", "sample_ids", "features", "training_log", "partitions"]:
            changed = copy.deepcopy(valid)
            del changed["rejection"][key]
            with self.subTest(key=key):
                self.assertTrue(expected_rejection_errors(changed))
        changed = copy.deepcopy(valid)
        changed["rejection"]["candidates"].pop()
        self.assertTrue(expected_rejection_errors(changed))
        self.train.write_text(self.train.read_text() + "\n")
        self.assertTrue(expected_rejection_errors(valid))

    def test_reports_distinguish_expected_rejection_from_training(self):
        result = self.verified()
        report = {"profile": "full", "finished_utc": "test", "results": [result]}
        self.assertEqual(write_reports(report, self.root), 0)
        coverage = report["coverage"]["matrix"]
        self.assertEqual((coverage["trained"], coverage["expected_rejections"], coverage["unexpected_failures"]), (0, 1, 0))
        suite = ET.parse(self.root / "junit.xml").getroot()
        self.assertEqual(suite.attrib["failures"], "0")
        self.assertEqual(suite.attrib["skipped"], "0")
        self.assertEqual(suite.find("testcase/properties/property[@name='outcome']").attrib["value"], "expected_rejection")
        self.assertIn("0 trained; 1 verified expected rejections", (self.root / "summary.md").read_text())
        report["profile"] = "quick"
        self.assertEqual(write_reports(report, self.root), 1)
        report["profile"] = "full"
        report["results"] = [result, copy.deepcopy(result)]
        self.assertEqual(write_reports(report, self.root), 1)

    def test_matrix_coverage_and_audit_allowance_remain_strict(self):
        result = self.verified()
        cases = [{"kind": "model", "dataset": d, "selection": s, "sampling": a, "model": m, "status": "passed"}
                 for d, s, a, m in itertools.product(["multiclass", "binary"], SELECTIONS, SAMPLINGS, MODELS)]
        self.assertEqual(expected_artifact_audits(cases, "full"), 246)
        position = next(i for i, r in enumerate(cases) if all(r[k] == result[k] for k in ["dataset", "selection", "sampling", "model"]))
        cases[position] = result
        self.assertEqual(coverage_errors(cases, "full"), [])
        self.assertEqual(expected_artifact_audits(cases, "full"), 245)
        self.assertTrue(coverage_errors(cases[:-1], "full"))
        self.assertTrue(coverage_errors(cases + [result], "full"))
        cases[-1] = {**cases[-1], "status": "failed", "error": "Missing artifact"}
        self.assertEqual(expected_artifact_audits(cases, "full"), 245)
        report = {"profile": "full", "finished_utc": "test", "results": [{**r, "stage": "matrix", "name": str(i)} for i, r in enumerate(cases)]}
        self.assertEqual(write_reports(report, self.root), 1)
        with self.assertRaises(ValueError):
            expected_artifact_audits([result, result], "full")

    def test_rejected_case_is_excluded_from_model_index(self):
        report = {"arguments": {"fixtures": str(ROOT / "test")}, "results": [self.verified()]}
        path = self.root / "index.tsv"
        export_model_index(report, path)
        self.assertEqual(len(path.read_text().splitlines()), 1)

    def test_successful_approved_case_keeps_normal_checks(self):
        from unittest.mock import patch
        runner = object.__new__(Runner)
        runner.results = []
        runner.report = {"results": runner.results}
        with patch.object(runner, "save"), patch.object(runner, "case", return_value={"artifact": "trained.joblib"}) as case:
            with patch("matrix.verify_enn_rejection") as verifier:
                result = runner.execute_case("multiclass", ROOT / "test/dataset1_multiclass", "muvr_RFC", {}, "XGBC", "enn")
        case.assert_called_once()
        verifier.assert_not_called()
        self.assertEqual(result["status"], "passed")

    def test_failed_case_dispatches_to_verified_rejection(self):
        from unittest.mock import patch
        runner = object.__new__(Runner)
        runner.output = self.root / "matrix"
        runner.results = []
        runner.report = {"results": runner.results, "commands": [self.command]}
        with patch.object(runner, "save"), patch.object(runner, "case", side_effect=ValueError("Command exited 1")):
            result = runner.execute_case("multiclass", ROOT / "test/dataset1_multiclass", "muvr_RFC",
                                         {"train": self.train}, "XGBC", "enn")
        self.assertEqual(result["status"], "expected_rejection")
        self.assertEqual(expected_rejection_errors(result), [])
        self.training_log.write_text("RuntimeError: unrelated\n")
        with patch.object(runner, "save"), patch.object(runner, "case", side_effect=ValueError("Command exited 1")):
            result = runner.execute_case("multiclass", ROOT / "test/dataset1_multiclass", "muvr_RFC",
                                         {"train": self.train}, "XGBC", "enn")
        self.assertEqual(result["status"], "failed")

    def test_stage_keeps_missing_artifact_audit_a_failure(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        valid = self.verified()
        validation = Validation("full", self.root / "validation")
        validation.report["results"] = [valid]
        child = self.root / "short_audit.json"
        child.write_text(json.dumps({"results": [{"status": "passed"} for _ in range(244)]}))
        with patch("validate.run_process", return_value=SimpleNamespace(returncode=0)):
            passed = validation.stage("artifact_audit", ["test-double"], child,
                                      expected_artifact_audits(validation.report["results"], "full"))
        self.assertFalse(passed)
        self.assertIn("Expected 245 results, found 244", validation.report["stages"][-1]["error"])

    def test_expected_evidence_is_preserved_separately(self):
        result = self.verified()
        matrix = self.root / "matrix"
        (matrix / "report.json").write_text(json.dumps({"results": [result]}))
        validation = object.__new__(Validation)
        validation.output = self.root
        validation.profile = "full"
        validation.report = {"results": [], "stages": [{"name": "matrix"}]}
        validation.stage_failed_artifacts()
        self.assertTrue((self.root / "expected_rejections/matrix/multiclass_muvr_RFC_XGBC_enn/expected_rejection.json").is_file())
        self.assertFalse((self.root / "failed_cases").exists())


if __name__ == "__main__":
    unittest.main()

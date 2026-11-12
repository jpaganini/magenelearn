"""Test the validator's ability to reject missing evidence and wrong probabilities."""
import itertools
import json
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
from matrix import check_artifact_probabilities, check_evaluation, frame
from validate import coverage_errors, write_reports, MODELS, SELECTIONS, Validation


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


if __name__ == "__main__":
    unittest.main()

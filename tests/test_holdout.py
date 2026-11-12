"""CLI orchestration regression tests; external scientific steps are test doubles."""
import importlib
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from click.testing import CliRunner

pipeline = importlib.import_module("maGeneLearn.cli")


def exercise(module, mode="generated"):
    events = []
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        root = Path(tmp)
        for name in ["meta.tsv", "features.tsv", "train.tsv", "test.tsv", "ready.tsv"]:
            (root / name).write_text("id\n")

        def split(ctx, meta):
            events.append("split")
            ctx.train_meta, ctx.test_meta = root / "train.tsv", root / "test.tsv"

        def extract(ctx):
            events.append("extract")
            ctx.feat_train = root / "ready.tsv"
            if mode != "no_holdout":
                ctx.feat_test = root / "ready.tsv"

        def run(cmd, **kwargs):
            events.append("holdout")
            if "--no_cv" not in cmd or str(root / "ready.tsv") not in cmd:
                raise AssertionError("Wrong hold-out evaluation command")

        for name, replacement in [
            ("split", split), ("extract_features", extract), ("run", run),
            ("chisq", lambda *a: events.append("chisq")),
            ("feature_selection", lambda *a: events.append("selection")),
            ("train_model", lambda *a: events.append("train")),
            ("evaluate_train", lambda *a: events.append("training_eval")),
        ]:
            stack.enter_context(patch.object(module, name, replacement))
        args = ["train", "--name", "regression", "--model", "LR", "--feature-model", "RFC",
                "--output-dir", str(root / "output")]
        if mode in {"generated", "selection_only"}:
            args += ["--meta-file", str(root / "meta.tsv"), "--features", str(root / "features.tsv"), "--chisq", "--muvr"]
        else:
            args += ["--train-meta", str(root / "train.tsv"), "--features-train", str(root / "ready.tsv")]
            if mode == "ready":
                args += ["--features-test", str(root / "ready.tsv")]
        if mode == "selection_only":
            args += ["--feature-selection-only"]
        result = CliRunner().invoke(module.cli, args)
        if result.exit_code:
            raise AssertionError(result.output) from result.exception
        has_holdout = (root / "output/07_test_eval").exists()
    return events, has_holdout


def require_generated_holdout(module):
    events, exists = exercise(module)
    if not exists or events != ["split", "chisq", "selection", "extract", "train", "training_eval", "holdout"]:
        raise AssertionError("Training omitted generated hold-out evaluation")


class HoldoutRegression(unittest.TestCase):
    def test_generated_holdout_is_evaluated_after_extraction(self):
        require_generated_holdout(pipeline)

    def test_supplied_ready_holdout_is_evaluated(self):
        events, exists = exercise(pipeline, "ready")
        self.assertEqual(events, ["extract", "train", "training_eval", "holdout"])
        self.assertTrue(exists)

    def test_no_holdout_does_not_create_evaluation(self):
        events, exists = exercise(pipeline, "no_holdout")
        self.assertEqual(events, ["extract", "train", "training_eval"])
        self.assertFalse(exists)

    def test_selection_only_does_not_train_or_evaluate(self):
        events, exists = exercise(pipeline, "selection_only")
        self.assertEqual(events, ["split", "chisq", "selection", "extract"])
        self.assertFalse(exists)

    def test_regression_detects_original_scheduling_in_memory(self):
        source = Path(pipeline.__file__).read_text()
        self.assertIn("(True, evaluate_holdout),", source)
        legacy = types.ModuleType("validation_legacy_cli")
        legacy.__file__ = pipeline.__file__
        with patch.dict(sys.modules, {legacy.__name__: legacy}):
            exec(compile(source.replace("(True, evaluate_holdout),", "(ctx.feat_test is not None, evaluate_holdout),"),
                         pipeline.__file__, "exec"), legacy.__dict__)
            with self.assertRaisesRegex(AssertionError, "omitted generated hold-out"):
                require_generated_holdout(legacy)


if __name__ == "__main__":
    unittest.main()

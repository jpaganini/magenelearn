"""Regression tests for class-preserving ENN neighbour fallback."""
import unittest

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import StratifiedKFold

from maGeneLearn.steps.sampling_validation import (
    SamplingClassLoss, check_sampling_classes, resolve_enn_neighbors,
)


class SamplingValidationTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(np.random.RandomState(0).normal(size=(36, 2)))
        self.y = np.repeat([0, 1, 2], [8, 12, 16])
        self.splits = list(StratifiedKFold(
            3, shuffle=True, random_state=50,
        ).split(self.X, self.y))

    def test_fallback_preserves_classes_in_every_partition(self):
        pipeline = Pipeline([("sampler", EditedNearestNeighbours(
            kind_sel="mode", n_neighbors=5,
        ))])
        original = self.X.copy(deep=True)
        with self.assertLogs(level="WARNING") as logs:
            actual = resolve_enn_neighbors(
                pipeline, self.X, self.y, self.splits, context="Regression",
            )
        self.assertEqual(actual, {"sampler__n_neighbors": 4})
        self.assertTrue(any("requested=5 | applied=4" in s for s in logs.output))
        for indices, _ in self.splits:
            check_sampling_classes(
                pipeline, self.X.iloc[indices], self.y[indices], context="Fold",
            )
        check_sampling_classes(pipeline, self.X, self.y, context="Final")
        self.assertFalse(hasattr(pipeline.named_steps["sampler"], "sample_indices_"))
        pd.testing.assert_frame_equal(self.X, original)

    def test_valid_setting_is_unchanged(self):
        pipeline = Pipeline([("sampler", EditedNearestNeighbours(
            kind_sel="mode", n_neighbors=4,
        ))])
        self.assertEqual(resolve_enn_neighbors(
            pipeline, self.X, self.y, self.splits, context="Valid",
        ), {"sampler__n_neighbors": 4})

    def test_exhausted_candidates_fail_without_disabling_sampler(self):
        X = pd.DataFrame(np.zeros((8, 2)))
        y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
        pipeline = Pipeline([("sampler", EditedNearestNeighbours(
            kind_sel="mode", n_neighbors=3,
        ))])
        with self.assertRaisesRegex(SamplingClassLoss, "Sampling was not disabled"):
            resolve_enn_neighbors(pipeline, X, y, [], context="Exhausted")
        self.assertEqual(pipeline.named_steps["sampler"].n_neighbors, 3)

    def test_smoteenn_keeps_smote_settings(self):
        pipeline = Pipeline([("sampler", SMOTEENN(
            smote=SMOTE(k_neighbors=1, random_state=50),
            enn=EditedNearestNeighbours(kind_sel="mode", n_neighbors=5),
            random_state=50,
        ))])
        actual = resolve_enn_neighbors(
            pipeline, self.X, self.y, self.splits, context="SMOTEENN",
        )
        self.assertIn("sampler__enn__n_neighbors", actual)
        self.assertEqual(pipeline.named_steps["sampler"].smote.k_neighbors, 1)
        check_sampling_classes(pipeline, self.X, self.y, context="Final")

    def test_no_sampler_is_unchanged(self):
        from sklearn.dummy import DummyClassifier
        pipeline = Pipeline([("model", DummyClassifier())])
        check_sampling_classes(pipeline, self.X, self.y, context="None")
        self.assertEqual(resolve_enn_neighbors(
            pipeline, self.X, self.y, self.splits, context="None",
        ), {})

    def test_optimization_and_final_fit_use_resolved_parameters(self):
        import importlib.util
        from pathlib import Path
        from unittest.mock import patch
        import optuna

        path = Path(__file__).resolve().parents[1] / "maGeneLearn/steps/04_train_model.py"
        spec = importlib.util.spec_from_file_location("training_sampling_test", path)
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        # Request five neighbours deterministically; the real sampler must
        # resolve four, and final fitting must use four rather than five.
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({"model__C": 1.0, "sampler__n_neighbors": 5})
        with patch.object(training.optuna, "create_study", return_value=study):
            fitted, history, parameters, _ = training.search_hyperparameters_optuna(
                training.prepare_pipeline("LR", "enn", 1),
                self.X, self.y, np.arange(len(self.y)), self.splits,
                "LR", "enn", 1, ["balanced_accuracy"], 1,
            )
        self.assertEqual(study.best_params["sampler__n_neighbors"], 5)
        self.assertEqual(parameters["sampler__n_neighbors"], 4)
        self.assertEqual(history.loc[0, "sampler__n_neighbors"], 4)
        self.assertEqual(fitted.named_steps["sampler"].n_neighbors, 4)
        self.assertEqual(set(fitted.classes_), set(self.y))

    def test_svm_final_fit_preserves_probability_support(self):
        import importlib.util
        from pathlib import Path
        from unittest.mock import patch
        import optuna

        path = Path(__file__).resolve().parents[1] / "maGeneLearn/steps/04_train_model.py"
        spec = importlib.util.spec_from_file_location("training_svm_test", path)
        training = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(training)
        study = optuna.create_study(direction="maximize")
        study.enqueue_trial({"model__C": 1.0, "model__kernel": "rbf",
                             "model__gamma": "scale", "model__class_weight": None})
        with patch.object(training.optuna, "create_study", return_value=study):
            fitted, history, _, _ = training.search_hyperparameters_optuna(
                training.prepare_pipeline("SVM", "none", 1),
                self.X, self.y, np.arange(len(self.y)), self.splits,
                "SVM", "none", 1, ["balanced_accuracy"], 1,
            )
        self.assertFalse(history.loc[0, "model__probability"])
        self.assertTrue(fitted.named_steps["model"].probability)
        probabilities = fitted.predict_proba(self.X)
        self.assertEqual(probabilities.shape, (len(self.y), 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)

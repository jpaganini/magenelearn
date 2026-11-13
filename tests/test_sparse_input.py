"""Sparse estimator/sampler compatibility and dense-zero scientific parity."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedGroupKFold
from unittest.mock import patch

from maGeneLearn.matrix_io import DenseZeroAdapter
from maGeneLearn.steps.sampling_validation import resolve_enn_neighbors


def load_step(filename):
    path = Path(__file__).resolve().parents[1] / "maGeneLearn/steps" / filename
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SparseInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_step("04_train_model.py")
        cls.evaluation = load_step("05_evaluate_model.py")

    def test_all_models_and_samplers_accept_csr_and_preserve_classes(self):
        for n_classes in [2, 3]:
            with self.subTest(n_classes=n_classes):
                self.check_models_samplers(n_classes)

    def check_models_samplers(self, n_classes):
        X, y = make_classification(n_samples=120, n_features=6, n_informative=4,
                                   n_redundant=0, class_sep=3, n_classes=n_classes,
                                   n_clusters_per_class=1, random_state=50)
        X[np.abs(X) < 1] = 0
        X = sparse.csr_matrix(X)
        groups = np.arange(len(y))
        folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=50).split(X, y, groups))
        for model in ["RFC", "XGBC", "SVM", "LR"]:
            for sampler in ["none", "random", "smote", "random_under", "enn", "smoteenn"]:
                with self.subTest(model=model, sampler=sampler):
                    pipeline = self.training.prepare_pipeline(model, sampler, 1)
                    if model == "XGBC":
                        pipeline.steps.insert(len(pipeline.steps) - 1, ("dense_zeros", DenseZeroAdapter()))
                        pipeline.set_params(model__n_estimators=5)
                    elif model == "RFC":
                        pipeline.set_params(model__n_estimators=5)
                    resolve_enn_neighbors(pipeline, X, y, folds, context="CSR regression")
                    estimator = pipeline.named_steps["model"]
                    with ExitStack() as stack:
                        fit = stack.enter_context(patch.object(estimator, "fit", wraps=estimator.fit))
                        resample = None
                        if "sampler" in pipeline.named_steps:
                            original_sampler = pipeline.named_steps["sampler"]
                            resample = stack.enter_context(patch.object(original_sampler, "fit_resample", wraps=original_sampler.fit_resample))
                        pipeline.fit(X, y)
                        self.assertEqual(sparse.issparse(fit.call_args[0][0]), model != "XGBC")
                        if resample is not None:
                            self.assertTrue(sparse.isspmatrix_csr(resample.call_args[0][0]),
                                            "Resampling must receive native CSR, including before the XGBoost dense adapter")
                    np.testing.assert_array_equal(pipeline.classes_, np.arange(n_classes))
                    p = pipeline.predict_proba(X)
                    self.assertEqual(p.shape, (120, n_classes))
                    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)

    def test_xgboost_dense_zero_parity_save_load_and_shuffled_features(self):
        X = np.array([[0, 1], [1, 0], [0, 2], [2, 0]] * 10, dtype=float)
        y = np.array([0, 1, 0, 1] * 10)
        dense = self.training.prepare_pipeline("XGBC", "none", 1)
        dense.set_params(model__n_estimators=5)
        csr = clone(dense)
        csr.steps.insert(0, ("dense_zeros", DenseZeroAdapter()))
        dense.fit(X, y)
        csr.fit(sparse.csr_matrix(X), y)
        np.testing.assert_allclose(dense.predict_proba(X), csr.predict_proba(sparse.csr_matrix(X)), atol=0)
        csr.mgl_feature_names = ["a", "b"]
        named = pd.DataFrame.sparse.from_spmatrix(sparse.csr_matrix(X), columns=["a", "b"])
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "model.joblib"
            joblib.dump(csr, artifact)
            restored = joblib.load(artifact)
            predictions, probabilities = self.evaluation.predict_with_pipeline(restored, named[["b", "a"]])
            np.testing.assert_array_equal(predictions, dense.predict(X))
            np.testing.assert_allclose(probabilities, dense.predict_proba(X))

    def test_legacy_xgboost_sparse_cv_preserves_dense_zero_semantics(self):
        from maGeneLearn.matrix_io import write_bundle
        from sklearn.preprocessing import LabelEncoder
        from sklearn.utils.class_weight import compute_sample_weight
        X = np.array([[0, 1], [1, 0], [0, 2], [2, 0]] * 10, dtype=float)
        y = np.array([0, 1, 0, 1] * 10)
        groups = np.arange(len(y))
        ids = [f"s{i}" for i in groups]
        encoder = LabelEncoder().fit(["A", "B"])
        pipeline = self.training.prepare_pipeline("XGBC", "none", 1)
        pipeline.set_params(model__n_estimators=5)
        pipeline.fit(pd.DataFrame(X, columns=["a", "b"]), y)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "legacy.joblib"
            joblib.dump({"model": pipeline, "label_encoder": encoder}, artifact)
            samples = pd.DataFrame({"label": encoder.inverse_transform(y), "group": groups}, index=ids)
            manifest = write_bundle(root / "input", sparse.csr_matrix(X), samples, ["a", "b"])
            self.evaluation.run_evaluation(artifact, manifest, "label", "group", 3, False,
                                           root / "result", "legacy", False, "balanced_accuracy",
                                           skip_shap=True, skip_svm_importance=True)
            predictions = pd.read_csv(root / "result/legacy_predictions_probabilities.tsv", sep="\t", index_col=0)
            for train, test in self.evaluation.get_cv_iterator(y, groups, 3):
                reference = clone(pipeline).fit(X[train], y[train],
                                               model__sample_weight=compute_sample_weight("balanced", y[train]))
                np.testing.assert_allclose(predictions.loc[[ids[i] for i in test], ["A", "B"]],
                                           reference.predict_proba(X[test]), atol=1e-6)

    def test_sparse_chisquare_matches_dense_presence_statistics_and_order(self):
        import subprocess
        import sys
        from maGeneLearn.matrix_io import write_bundle, read_bundle
        from sklearn.feature_selection import chi2
        chi = load_step("01_chisq_selection.py")
        # Counts >127, negatives, explicit zeros, and a tied ranking exercise
        # presence semantics independently of the pipeline's binary fixtures.
        X = np.array([[300, 0, 0, -1, 900], [500, 0, 1, 0, 900],
                      [0, 200, 1, 0, 0], [0, 400, 0, 0, 0]])
        ids = ["s1", "s2", "s3", "s4"]
        names = ["b", "a", "c", "zero", "Unnamed:0"]
        labels = ["A", "A", "B", "B"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "metadata.tsv"
            pd.DataFrame({"label": labels}, index=ids).to_csv(meta, sep="\t")
            manifest = write_bundle(root / "input", sparse.csr_matrix(X), pd.DataFrame(index=ids), names)
            completed = subprocess.run([sys.executable, str(Path(chi.__file__)), "--meta", str(meta),
                                       "--features1", str(manifest), "--label", "label", "--k", "2",
                                       "--output_dir", str(root / "result"), "--name", "test"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            pvalues = pd.read_csv(root / "result/test_pvalues.tsv", sep="\t")
            _, expected = chi2((X[:, :4] > 0).astype(int), labels)
            np.testing.assert_allclose(pvalues.iloc[:, 1], expected, equal_nan=True, rtol=1e-6)
            selected, samples, features, info = read_bundle(root / "result/test_top2_features.csr")
            self.assertEqual(features.tolist(), ["b", "a"])
            self.assertEqual(samples.index.tolist(), ids)
            np.testing.assert_array_equal(selected.toarray(), (X[:, :2] > 0).astype(int))


if __name__ == "__main__":
    unittest.main()

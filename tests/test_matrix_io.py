"""CSR format and conversion invariants, including value/identifier integrity."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from maGeneLearn.matrix_io import (DenseZeroAdapter, as_csr, dense_input, read_bundle,
                                   read_table, select_columns, stream_tsv, write_bundle)


class MatrixIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.X = sparse.csr_matrix([[0, 300, 0], [2, 0, 1], [0, 0, 0]], dtype=np.int64)
        self.samples = pd.DataFrame({"label": ["A", "B", "A"], "group": ["01", "02", "03"]},
                                    index=["001", "NA", "s3"])
        self.names = ["f3", "f1", "f2"]

    def test_portable_bundle_preserves_counts_and_axes(self):
        manifest = write_bundle(self.root / "input.csr", self.X, self.samples, self.names)
        renamed = self.root / "moved.csr"
        manifest.parent.rename(renamed)
        restored, samples, names, info = read_bundle(renamed)
        np.testing.assert_array_equal(restored.toarray(), self.X.toarray())
        pd.testing.assert_frame_equal(samples, self.samples, check_names=False)
        self.assertEqual(names.tolist(), self.names)
        table = read_table(renamed)
        np.testing.assert_array_equal(as_csr(table[self.names]).toarray(), self.X.toarray())
        with self.assertRaises(FileExistsError):
            write_bundle(renamed, self.X, self.samples, self.names)

    def test_corruption_and_axis_mismatch_fail(self):
        manifest = write_bundle(self.root / "input.csr", self.X, self.samples, self.names)
        (manifest.parent / "features.tsv").write_text("feature_id\nchanged\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_bundle(manifest)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            write_bundle(self.root / "bad", self.X[:, :2], self.samples, self.names)

    def test_duplicate_or_empty_axes_fail(self):
        for names in [["f", " f ", "z"], ["f", "", "z"]]:
            with self.assertRaisesRegex(ValueError, "identifiers"):
                write_bundle(self.root / "bad", self.X, self.samples, names)

    def test_noncanonical_coordinates_and_nonfinite_values_fail(self):
        duplicate = sparse.csr_matrix(([1, 2], [0, 0], [0, 2]), shape=(1, 1))
        with self.assertRaisesRegex(ValueError, "duplicate coordinates"):
            write_bundle(self.root / "bad", duplicate, pd.DataFrame(index=["s"]), ["f"])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            write_bundle(self.root / "bad", sparse.csr_matrix([[np.nan]]), pd.DataFrame(index=["s"]), ["f"])

    def test_sparse_column_alignment_and_zero_fill(self):
        selected = select_columns(self.X, self.names, ["f2", "absent", "f3"])
        np.testing.assert_array_equal(selected.toarray(), [[0, 0, 0], [1, 0, 2], [0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "None of"):
            select_columns(self.X, self.names, ["absent"])

    def test_projection_is_canonical_and_mixed_columns_stay_sparse(self):
        projected = select_columns(sparse.csr_matrix([[1, 2], [3, 4]]), ["a", "b"], ["a", "b"])
        self.assertTrue(projected.has_canonical_format)
        manifest = write_bundle(self.root / "projection", projected, pd.DataFrame(index=["s1", "s2"]), ["a", "b"])
        np.testing.assert_array_equal(read_bundle(manifest)[0].toarray(), [[1, 2], [3, 4]])
        table = pd.DataFrame.sparse.from_spmatrix(self.X, columns=self.names)
        table["missing"] = 0
        with patch.object(pd.DataFrame, "to_numpy", side_effect=AssertionError("Whole-table densification")):
            restored = as_csr(table)
        self.assertTrue(sparse.isspmatrix_csr(restored))
        np.testing.assert_array_equal(restored.toarray(), np.column_stack([self.X.toarray(), [0, 0, 0]]))

    def test_stream_assessment_and_conversion_do_not_binarize(self):
        path = self.root / "matrix.tsv"
        path.write_text("ID\tf1\tf2\tlabel\n001\t300\t0\tA\nNA\t0\t1.5\tB\n")
        original = path.read_bytes()
        empty, samples, names, report = stream_tsv(path, metadata=["label"])
        self.assertIsNone(empty)
        X, samples, names, second = stream_tsv(path, metadata=["label"], convert=True)
        self.assertEqual(report, second)
        np.testing.assert_array_equal(X.toarray(), [[300, 0], [0, 1.5]])
        self.assertEqual(samples.index.tolist(), ["001", "NA"])
        self.assertEqual(path.read_bytes(), original)

    def test_stream_rejects_malformed_missing_and_lossy_values(self):
        for body in ["s\t1\n", "s\tNaN\t0\n", "s\t9007199254740993\t0\n", "s\t1\t0\ns\t0\t1\n"]:
            path = self.root / "invalid.tsv"
            path.write_text("ID\tf1\tf2\n" + body)
            with self.assertRaises(ValueError):
                stream_tsv(path, convert=True)

    def test_dense_memory_guard_and_zero_adapter(self):
        with patch.dict("os.environ", {"MAGENELEARN_MAX_DENSE_MB": "0.000001"}):
            with self.assertRaises(MemoryError):
                dense_input(self.X)
        np.testing.assert_array_equal(DenseZeroAdapter().fit_transform(self.X), self.X.toarray())

    def test_assessment_cli_never_auto_always_and_existing_output(self):
        path = Path(__file__).resolve().parents[1] / "scripts/prepare_sparse_input.py"
        spec = importlib.util.spec_from_file_location("prepare_sparse_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = self.root / "source.tsv"
        source.write_text("ID\tf1\tf2\ns1\t1\t1\ns2\t1\t1\n")
        for mode, converted in [("never", False), ("auto", False), ("always", True)]:
            destination = self.root / mode
            with patch("sys.argv", ["prepare", "--input", str(source), "--output", str(destination), "--convert", mode]):
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            self.assertEqual(json.loads((destination / "assessment.json").read_text())["converted"], converted)
            self.assertEqual((destination / "manifest.json").exists(), converted)
        with patch("sys.argv", ["prepare", "--input", str(source), "--output", str(destination)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                module.main()

    def test_numeric_metadata_types_survive_round_trip(self):
        samples = pd.DataFrame({"label": [1, 2, 1], "group": [11, 12, 13]}, index=self.samples.index)
        manifest = write_bundle(self.root / "numeric", self.X, samples, self.names)
        pd.testing.assert_frame_equal(read_bundle(manifest)[1], samples, check_names=False)

    def test_integer_dtype_preserves_smote_generation(self):
        from imblearn.over_sampling import SMOTE
        source = self.root / "integer.tsv"
        source.write_text("ID\tf1\ns1\t0\ns2\t0\ns3\t0\ns4\t1\ns5\t4\n")
        X, _, _, report = stream_tsv(source, convert=True)
        self.assertEqual(X.dtype, np.dtype("int64"))
        self.assertEqual(report["dtype"], "int64")
        original = pd.read_csv(source, sep="\t", index_col=0)
        y = np.array([0, 0, 0, 1, 1])
        dense, dense_y = SMOTE(k_neighbors=1, random_state=50).fit_resample(original, y)
        csr, csr_y = SMOTE(k_neighbors=1, random_state=50).fit_resample(X, y)
        np.testing.assert_array_equal(csr.toarray(), dense)
        np.testing.assert_array_equal(csr_y, dense_y)
        source.write_text(source.read_text().replace("s5\t4", "s5\t4.0"))
        self.assertEqual(stream_tsv(source, convert=True)[0].dtype, np.dtype("float64"))


if __name__ == "__main__":
    unittest.main()

"""Portable integrity checks on bundled fixtures, never on local source uploads."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test"


def table(path):
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    header = rows.pop(0)
    if len(header) != len(set(header)) or any(len(r) != len(header) for r in rows):
        raise ValueError("Duplicate columns or malformed row: " + str(path))
    if len({r[0] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample ID: " + str(path))
    return header, rows


class FixtureIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((FIXTURES / "manifest.json").read_text())
        cls.samples = cls.manifest["samples"]
        cls.names = cls.manifest["feature_order"]

    def test_bundled_checksums(self):
        for rel, expected in self.manifest["files"].items():
            with self.subTest(file=rel):
                self.assertEqual(hashlib.sha256((FIXTURES / rel).read_bytes()).hexdigest(), expected)
        # Original source paths are provenance, not runtime dependencies.
        self.assertTrue(self.manifest["inputs"])

    def test_membership_quotas_and_groups(self):
        self.assertEqual(len(self.samples), 210)
        self.assertEqual(len({r["Sample.ID"] for r in self.samples}), 210)
        self.assertEqual(len({r["cluster5"] for r in self.samples}), 210)
        header, rows = table(FIXTURES / "membership.tsv")
        self.assertEqual([dict(zip(header, r)) for r in rows], self.samples)
        for source, count in {"Bovine": 40, "Poultry": 50, "Swine": 60}.items():
            for part, expected in [("train", count), ("holdout", 10), ("external", 10)]:
                subset = [r for r in self.samples if r["source_summary"] == source and r["partition"] == part]
                self.assertEqual(len(subset), expected)
                self.assertEqual(collections.Counter(r["plot_MLST"] for r in subset),
                                 {"19": expected // 2, "34": expected // 2})
                self.assertTrue(all(r["cluster5"] not in {"", "NA", "-1"} for r in subset))

    def test_schema_values_labels_and_partitions(self):
        for dataset, binary, external in [("dataset1_multiclass", False, False), ("dataset2_binary", True, False),
                                           ("dataset3_test/multiclass", False, True), ("dataset3_test/binary", True, True)]:
            directory = FIXTURES / dataset
            expected = {r["Sample.ID"]: r for r in self.samples
                        if (r["partition"] == "external") == external
                        and (not binary or r["source_summary"] != "Poultry")}
            h, rows = table(directory / "metadata.tsv")
            self.assertEqual(h, ["Sample.ID", "source_summary", "cluster5", "plot_MLST"])
            self.assertEqual({r[0] for r in rows}, set(expected))
            for row in rows:
                self.assertEqual(row, [expected[row[0]][c] for c in h])
            h, features = table(directory / "features.tsv")
            self.assertEqual(h, ["Sample.ID"] + self.names)
            self.assertEqual([r[0] for r in features], [r[0] for r in rows])
            self.assertTrue(all(v in {"0", "1"} for row in features for v in row[1:]))
            by_id = {r[0]: r[1:] for r in features}
            paths = ["ready_labelled.tsv"] if external else ["ready_train.tsv", "ready_holdout.tsv"]
            for path in paths:
                header, ready = table(directory / path)
                self.assertEqual(header, ["Sample.ID", "source_summary", "cluster5"] + self.names)
                want = {s for s, r in expected.items() if external or r["partition"] == path[6:-4]}
                self.assertEqual({r[0] for r in ready}, want)
                for row in ready:
                    self.assertEqual(row[1:3], [expected[row[0]]["source_summary"], expected[row[0]]["cluster5"]])
                    self.assertEqual(row[3:], by_id[row[0]])
            if external:
                self.assertEqual(table(directory / "predict_features.tsv"), (h, features))
            else:
                for part in ["train", "holdout"]:
                    _, partition = table(directory / (part + "_metadata.tsv"))
                    self.assertEqual({r[0] for r in partition}, {s for s, r in expected.items() if r["partition"] == part})
            self.assertEqual(table(directory / "all_features.tsv"), (["Sample.ID"] + self.names, []))

    def test_no_repeated_profiles_and_constants(self):
        _, development = table(FIXTURES / "dataset1_multiclass/features.tsv")
        _, reserved = table(FIXTURES / "dataset3_test/multiclass/features.tsv")
        self.assertEqual(len({tuple(r[1:]) for r in development + reserved}), 210)
        _, train = table(FIXTURES / "dataset1_multiclass/ready_train.tsv")
        self.assertEqual(sum(len({r[j] for r in train}) == 1 for j in range(3, len(train[0]))), 451)
        self.assertEqual(len(self.names), 2471)

    def test_binary_predictors_equal_multiclass_subset(self):
        for multi, binary in [("dataset1_multiclass", "dataset2_binary"),
                              ("dataset3_test/multiclass", "dataset3_test/binary")]:
            _, rows = table(FIXTURES / multi / "features.tsv")
            by_id = {r[0]: r for r in rows}
            _, subset = table(FIXTURES / binary / "features.tsv")
            self.assertTrue(all(row == by_id[row[0]] for row in subset))

    def test_deduplication_companion(self):
        _, rows = table(FIXTURES / "selector_dedup_case/metadata.tsv")
        self.assertEqual(len(rows), 6)
        retained = {}
        for row in sorted(rows):
            retained[tuple(row[1:3])] = row[0]
        _, expected = table(FIXTURES / "selector_dedup_case/expected_retained.tsv")
        self.assertEqual(sorted(retained.values()), [r[0] for r in expected])
        self.assertEqual(len(expected), 3)


if __name__ == "__main__":
    unittest.main()

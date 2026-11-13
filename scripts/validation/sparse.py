"""Run the declared model matrix using fresh CSR inputs and outputs."""
from pathlib import Path

from common import checksum, matrix_path
from matrix import Runner, frame


class SparseRunner(Runner):
    def command(self, argv, log):
        return super().command(sparse_arguments(argv, self.output / "csr_inputs"), log)


def sparse_arguments(argv, directory):
    from maGeneLearn.matrix_io import is_bundle, write_table_bundle, read_bundle
    argv = list(map(str, argv))
    flags = {"--features", "--features-train", "--features-test", "--feature-file", "--chisq-file", "--chisq_file"}
    for index, flag in enumerate(argv[:-1]):
        if flag not in flags:
            continue
        path = matrix_path(argv[index + 1])
        if not path.exists():
            continue  # Invalid-input tests must reach the CLI's own validation.
        if not is_bundle(path):
            destination = directory / checksum(path)
            if not destination.exists():
                table = frame(path)
                if table.empty:
                    continue  # Legacy header-only feature specifications remain supported.
                metadata = [c for c in ["source_summary", "cluster5"] if c in table]
                provenance = {"source": str(path), "sha256": checksum(path)}
                if flag in {"--chisq_file", "--chisq-file"}:
                    provenance["training_ids"] = table.index.tolist()
                write_table_bundle(destination, table, metadata, provenance=provenance)
            path = destination / "manifest.json"
        read_bundle(path)
        argv[index + 1] = str(path)
    return argv

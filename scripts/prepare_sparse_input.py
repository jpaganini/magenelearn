#!/usr/bin/env python3
"""Assess a TSV/CSR feature matrix and optionally create a reusable CSR bundle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maGeneLearn.matrix_io import (checksum, is_bundle, read_bundle, stream_tsv, write_bundle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory; required for conversion")
    parser.add_argument("--convert", choices=["never", "auto", "always"], default="never")
    parser.add_argument("--metadata-columns", nargs="*", default=[], help="Non-feature columns embedded in the TSV")
    parser.add_argument("--min-savings", type=float, default=0.2, help="Minimum estimated CSR RAM saving for auto (default 0.2)")
    parser.add_argument("--meta-file", type=Path, help="Optional separate metadata TSV for overlap reporting")
    args = parser.parse_args()
    if not 0 <= args.min_savings < 1:
        parser.error("--min-savings must be in [0, 1)")
    if args.output and args.output.exists():
        parser.error("--output must be a new directory")
    if args.convert != "never" and args.output is None:
        parser.error("--output is required for conversion")
    if is_bundle(args.input):
        import numpy as np
        X, samples, features, info = read_bundle(args.input)
        dense_bytes = X.shape[0] * X.shape[1] * X.dtype.itemsize
        report = {"shape": list(X.shape), "dtype": str(X.dtype), "nnz": X.nnz,
                  "density": X.nnz / (X.shape[0] * X.shape[1]), "dense_bytes": dense_bytes,
                  "csr_bytes": X.data.nbytes + X.indices.nbytes + X.indptr.nbytes,
                  "already_sparse": True, "manifest": info,
                  "zero_rows": int((np.diff(X.indptr) == 0).sum()),
                  "zero_columns": int((np.bincount(X.indices, minlength=X.shape[1]) == 0).sum()),
                  "negative_values": int((X.data < 0).sum()),
                  "fractional_values": int((X.data != np.trunc(X.data)).sum())}
        minimum = float(X.data.min()) if X.nnz else 0.0
        maximum = float(X.data.max()) if X.nnz else 0.0
        if X.nnz < X.shape[0] * X.shape[1]:
            minimum, maximum = min(0.0, minimum), max(0.0, maximum)
        report.update(minimum=minimum, maximum=maximum)
    else:
        X, samples, features, report = stream_tsv(args.input, metadata=args.metadata_columns)
    report["recommended_sparse"] = report["csr_bytes"] <= report["dense_bytes"] * (1 - args.min_savings)
    if args.meta_file:
        import pandas as pd
        from maGeneLearn.matrix_io import identifiers
        meta = pd.read_csv(args.meta_file, sep="\t", dtype=str, keep_default_na=False, index_col=0)
        meta.index = identifiers(meta.index, "Metadata sample")
        report["metadata_overlap"] = {"shared": len(samples.index.intersection(meta.index)),
                                      "matrix_only": samples.index.difference(meta.index).tolist(),
                                      "metadata_only": meta.index.difference(samples.index).tolist()}
    convert = args.convert == "always" or args.convert == "auto" and report["recommended_sparse"]
    if convert:
        if X is None:
            X, samples, features, second = stream_tsv(args.input, metadata=args.metadata_columns, convert=True)
            if second != {k: report[k] for k in second}:
                raise ValueError("Input changed during assessment/conversion")
        provenance = {"source": str(args.input.resolve()), "assessment": report}
        write_bundle(args.output, X, samples, features, provenance=provenance)
        # Verify the persisted matrix and both ordered axes, not merely the exit status.
        restored, saved_samples, saved_features, _ = read_bundle(args.output)
        if ((restored != X).nnz or not saved_features.equals(features)
                or not saved_samples.astype(str).equals(samples.astype(str))):
            raise ValueError("Sparse conversion round-trip mismatch")
    elif args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    report["converted"] = convert
    if args.output:
        (args.output / "assessment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

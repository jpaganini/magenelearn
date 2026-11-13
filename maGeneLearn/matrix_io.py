"""Portable CSR bundles, identifier alignment and explicit dense boundaries.

Implicit entries always mean numeric zero, never missing observations.
"""
import csv
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_bundle(path):
    path = Path(path)
    return path.is_dir() or path.suffix == ".json"


def manifest_path(path):
    path = Path(path)
    return path / "manifest.json" if path.is_dir() else path


def identifiers(values, kind):
    values = list(values)
    if any(pd.isna(value) for value in values):
        raise ValueError(f"{kind} identifiers must not be missing")
    result = pd.Index([str(value).strip() for value in values])
    if result.empty or any(not value for value in result) or not result.is_unique:
        raise ValueError(f"{kind} identifiers must be nonempty and unique after stripping whitespace")
    return result


def as_csr(X):
    if sparse.issparse(X):
        return X.tocsr()
    if isinstance(X, pd.DataFrame) and all(isinstance(t, pd.SparseDtype) for t in X.dtypes):
        return X.sparse.to_coo().tocsr()
    if isinstance(X, pd.DataFrame) and any(isinstance(t, pd.SparseDtype) for t in X.dtypes):
        # Reindexing legacy feature tables can add dense zero-filled columns.
        # Convert only those columns, rather than densifying the entire table.
        columns = X.copy(deep=False)
        for name in columns:
            if not isinstance(columns[name].dtype, pd.SparseDtype):
                columns[name] = pd.arrays.SparseArray(columns[name].to_numpy(), fill_value=0)
        return columns.sparse.to_coo().tocsr()
    return sparse.csr_matrix(X)


def is_sparse_frame(X):
    return isinstance(X, pd.DataFrame) and any(isinstance(t, pd.SparseDtype) for t in X.dtypes)


def model_input(X):
    return as_csr(X) if is_sparse_frame(X) else X


def dense_input(X, *, context="dense adapter", multiplier=1):
    """Limit each dense allocation; the limit is not a total process-RAM budget."""
    if not sparse.issparse(X) and not is_sparse_frame(X):
        return X
    X = as_csr(X)
    limit = float(os.environ.get("MAGENELEARN_MAX_DENSE_MB", "1024")) * 1024 ** 2
    needed = X.shape[0] * X.shape[1] * X.dtype.itemsize * multiplier
    if not np.isfinite(limit) or limit <= 0 or needed > limit:
        raise MemoryError(f"{context}: estimated dense workspace {needed / 1024**2:.1f} MiB exceeds "
                          "MAGENELEARN_MAX_DENSE_MB (default 1024). Reduce the input or explicitly raise the limit.")
    logging.info("%s: dense workspace estimate %.1f MiB", context, needed / 1024**2)
    return X.toarray()


class DenseZeroAdapter(TransformerMixin, BaseEstimator):
    """Preserve dense-zero XGBoost semantics rather than sparse-missing semantics."""
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return dense_input(X, context="XGBoost zero-preserving adapter")


def write_bundle(path, X, samples, features, *, provenance=None):
    path = Path(path)
    X = as_csr(X)
    if X.dtype.kind not in "biuf":
        raise ValueError("Feature matrix must contain real numeric values")
    X.check_format(full_check=True)
    if not X.has_canonical_format:
        raise ValueError("CSR must have sorted indices and no duplicate coordinates")
    if np.any(X.data == 0):
        X = X.copy()
        X.eliminate_zeros()
    samples = samples.copy()
    samples.index = identifiers(samples.index, "Sample")
    features = identifiers(features, "Feature")
    if X.shape != (len(samples), len(features)) or not np.isfinite(X.data).all():
        raise ValueError("Matrix dimensions/identifiers disagree or matrix contains nonfinite values")
    if set(samples.columns) & set(features):
        raise ValueError("Feature identifiers collide with metadata columns")
    path.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(path / "matrix.npz", X)
    samples.to_csv(path / "samples.tsv", sep="\t", index_label="sample_id")
    pd.DataFrame({"feature_id": features}).to_csv(path / "features.tsv", sep="\t", index=False)
    manifest = {"format": "maGeneLearn.csr", "version": 1, "shape": list(X.shape),
                "dtype": str(X.dtype), "nnz": int(X.nnz), "implicit_value": 0,
                "metadata_dtypes": {c: str(t) for c, t in samples.dtypes.items()},
                "files": {name: checksum(path / name) for name in ["matrix.npz", "samples.tsv", "features.tsv"]},
                "provenance": provenance or {}}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path / "manifest.json"


def read_bundle(path):
    manifest = manifest_path(path)
    info = json.loads(manifest.read_text())
    if (info.get("format"), info.get("version"), info.get("implicit_value")) != ("maGeneLearn.csr", 1, 0):
        raise ValueError("Unsupported sparse manifest format/version/implicit value")
    for name in ["matrix.npz", "samples.tsv", "features.tsv"]:
        if info.get("files", {}).get(name) != checksum(manifest.parent / name):
            raise ValueError(f"Sparse bundle checksum mismatch: {name}")
    X = sparse.load_npz(manifest.parent / "matrix.npz")
    if not sparse.isspmatrix_csr(X) or X.dtype.kind not in "biuf":
        raise ValueError("Bundle matrix must be CSR")
    X.check_format(full_check=True)
    if not X.has_canonical_format or not np.isfinite(X.data).all():
        raise ValueError("CSR must be canonical and contain finite numeric values")
    samples = pd.read_csv(manifest.parent / "samples.tsv", sep="\t", dtype=str, keep_default_na=False, index_col=0)
    samples.index = identifiers(samples.index, "Sample")
    for column, dtype in info.get("metadata_dtypes", {}).items():
        if column not in samples:
            raise ValueError(f"Missing metadata column: {column}")
        if dtype not in {"object", "str", "string", "category"}:
            if dtype == "bool":
                if not samples[column].isin(["True", "False"]).all():
                    raise ValueError(f"Invalid boolean metadata: {column}")
                samples[column] = samples[column] == "True"
            else:
                samples[column] = samples[column].astype(dtype)
    features = identifiers(pd.read_csv(manifest.parent / "features.tsv", sep="\t", dtype=str,
                                      keep_default_na=False)["feature_id"], "Feature")
    if (list(X.shape) != info["shape"] or X.shape != (len(samples), len(features))
            or str(X.dtype) != info["dtype"] or X.nnz != info["nnz"]):
        raise ValueError("Sparse bundle shape/dtype/nnz disagrees with manifest")
    if set(samples.columns) & set(features):
        raise ValueError("Feature identifiers collide with metadata columns")
    return X, samples, features, info


def read_table(path):
    """Use sparse pandas columns only where existing reporting needs named axes."""
    if not is_bundle(path):
        if Path(path).suffix == ".npz":
            raise ValueError("Use a CSR bundle manifest.json; a bare .npz has no sample/feature identifiers")
        with Path(path).open(newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"))
        identifiers(header[1:], "Column")
        table = pd.read_csv(path, sep="\t", index_col=0, dtype={header[0]: str})
        table.index = identifiers(table.index, "Sample")
        return table
    X, samples, features, _ = read_bundle(path)
    table = pd.DataFrame.sparse.from_spmatrix(X, index=samples.index, columns=features)
    return pd.concat([table, samples], axis=1)


def write_table_bundle(path, table, metadata=(), *, provenance=None):
    metadata = [c for c in metadata if c in table]
    return write_bundle(path, as_csr(table.drop(columns=metadata)), table[metadata],
                        table.drop(columns=metadata).columns, provenance=provenance)


def select_columns(X, names, requested):
    names = identifiers(names, "Feature")
    requested = identifiers(requested, "Selected feature")
    positions = names.get_indexer(requested)
    found = positions >= 0
    if not found.any():
        raise ValueError("None of the selected features are present in the full matrix")
    if not found.all():
        logging.warning("Zero-filling %d missing selected features: %s", (~found).sum(), requested[~found][:5].tolist())
    # Remap existing columns through a sparse projection; no dense zero columns.
    projection = sparse.csr_matrix((np.ones(found.sum(), dtype=X.dtype),
                                    (positions[found], np.flatnonzero(found))),
                                   shape=(len(names), len(requested)))
    result = (X @ projection).tocsr()
    result.sort_indices()
    return result


def stream_tsv(path, *, metadata=(), convert=False):
    """Assess in one row-wise pass; optionally build CSR with O(nnz) storage.

    Preserve integer versus floating input storage because SMOTE uses input
    dtype when generating samples. Integers outside float64's exact range are
    rejected rather than silently rounded. Conversion never binarizes.
    """
    from array import array
    metadata = list(metadata)
    data, indices, indptr = array("q"), array("q"), array("q", [0])
    integer_input = True
    ids, sample_rows = [], []
    seen = set()
    nnz = zero_rows = negative = fractional = 0
    minimum = maximum = None
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        columns = identifiers(header[1:], "Column")
        missing = set(metadata) - set(columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")
        positions = [i for i, name in enumerate(columns) if name not in metadata]
        features = identifiers(columns[positions], "Feature")
        meta_positions = [columns.get_loc(c) for c in metadata]
        feature_seen = np.zeros(len(features), dtype=bool)
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Row {line}: expected {len(header)} fields, found {len(row)}")
            sample = row[0].strip()
            if not sample or sample in seen:
                raise ValueError(f"Row {line}: missing or duplicate sample ID {sample!r}")
            seen.add(sample)
            ids.append(sample)
            sample_rows.append([row[i + 1] for i in meta_positions])
            tokens = [row[i + 1] for i in positions]
            if integer_input and not all(token.strip().lstrip("+-").isdigit() for token in tokens):
                integer_input = False
                if convert:
                    data = array("d", data)
            try:
                values = np.asarray(tokens, dtype=np.float64)
            except ValueError as error:
                raise ValueError(f"Row {line}: nonnumeric feature value; declare label/group columns with --metadata-columns") from error
            if not np.isfinite(values).all():
                raise ValueError(f"Row {line}: missing/nonfinite feature value")
            # Decimal checks prevent loss of large integers or over-precise decimals.
            from decimal import Decimal
            for token, value in zip(tokens, values):
                if Decimal(token) != Decimal(str(value)):
                    raise ValueError(f"Row {line}: value {token!r} cannot be represented losslessly as float64")
                if Decimal(token) == Decimal(token).to_integral_value() and abs(Decimal(token)) > 2 ** 53:
                    raise ValueError(f"Row {line}: integer outside exact float64 range")
            present = np.flatnonzero(values)
            nnz += len(present)
            zero_rows += not len(present)
            negative += int((values < 0).sum())
            fractional += int((values != np.trunc(values)).sum())
            feature_seen[present] = True
            minimum = min(minimum, values.min()) if minimum is not None else values.min()
            maximum = max(maximum, values.max()) if maximum is not None else values.max()
            if convert:
                data.extend(values[present].astype(np.int64) if integer_input else values[present])
                indices.extend(present)
                indptr.append(len(data))
    if not ids:
        raise ValueError("Feature matrix has no samples")
    shape = (len(ids), len(features))
    index_bytes = 4 if max(nnz, *shape) <= np.iinfo(np.int32).max else 8
    dense_bytes = shape[0] * shape[1] * 8
    csr_bytes = nnz * (8 + index_bytes) + (shape[0] + 1) * index_bytes
    dtype = np.int64 if integer_input else np.float64
    report = {"shape": list(shape), "dtype": np.dtype(dtype).name, "nnz": nnz,
              "density": nnz / (shape[0] * shape[1]), "dense_bytes": dense_bytes,
              "csr_bytes": csr_bytes, "zero_rows": zero_rows, "zero_columns": int((~feature_seen).sum()),
              "negative_values": negative, "minimum": float(minimum), "maximum": float(maximum),
              "fractional_values": fractional,
              "source_sha256": checksum(path)}
    X = sparse.csr_matrix((np.frombuffer(data, dtype=dtype),
                           np.frombuffer(indices, dtype=np.int64), np.frombuffer(indptr, dtype=np.int64)),
                          shape=shape) if convert else None
    return X, pd.DataFrame(sample_rows, index=ids, columns=metadata), features, report

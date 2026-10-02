#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_chisq_selection_streaming.py
===============================

Chi-squared feature selection for extremely wide genomic TSV matrices.

This version replaces repeated pandas column-block scans with row-wise
streaming. It is designed for matrices with millions of feature columns.

Inputs
------
--meta
    Metadata TSV. First column = sample ID.

--features1
    Feature matrix TSV:
        rows    = isolates
        columns = features
        first column = sample IDs

--output_dir
    Directory for outputs.

--name
    Base output name.

Parameters
----------
--label
    Metadata label column (default: SYMP).

--k
    Number of top Chi² features to retain (default: 100000).

--n_jobs
    Accepted for backwards CLI compatibility but not used by the streaming
    implementation.

--block-cols
    Accepted for backwards CLI compatibility but not used by the streaming
    implementation.

Outputs
-------
<name>_pvalues.tsv
    feature    p_value

<name>_top{k}_features.tsv
    Dense binary TSV containing the selected top-k features.

Key change
----------
The original implementation reread the complete ultra-wide TSV once for every
feature block. This version performs:

    Pass A:
        ONE row-wise scan of the source TSV to calculate Chi² sufficient
        statistics (per-class feature-presence counts).

    Pass B:
        ONE row-wise scan of the source TSV to extract the selected top-k
        features and write the final matrix.

Thus, the source feature matrix is scanned only twice in total.

The feature values are binarized exactly as in the previous implementation:

    value > 0  ->  1
    otherwise  ->  0

Chi² equivalence
----------------
For non-negative X, sklearn.feature_selection.chi2 computes:

    observed[class, feature]
        = sum of X for that feature among samples in that class

For the binary matrix used here, that is simply the number of samples in each
class in which the feature is present.

The expected values are:

    class_probability * total_feature_count

and the Chi² statistic is calculated from those observed/expected counts.
This script reproduces that calculation directly without constructing the
full samples × features matrix in memory.

IMPORTANT
---------
The entire raw feature header is read once as a byte line so feature names can
be recovered. It is NOT parsed by pandas.

Rows are parsed numerically with numpy.fromstring(), one sample at a time.
"""

import argparse
import logging
import os
import sys
import time
import threading

from datetime import datetime
from heapq import heappush, heappop
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import psutil

from scipy import special
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def positive_int(value: str) -> int:
    value = int(value)

    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")

    return value


def get_opts():

    parser = argparse.ArgumentParser(
        description=(
            "Chi² feature selection from extremely wide genomic TSV data "
            "using row-wise streaming."
        )
    )

    parser.add_argument(
        "--meta",
        required=True,
        help="Path to metadata TSV with sample IDs in the first column."
    )

    parser.add_argument(
        "--features1",
        required=True,
        help=(
            "Path to feature matrix TSV; rows = isolates, columns = features, "
            "first column = sample IDs."
        )
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output files."
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Base name for output files."
    )

    parser.add_argument(
        "--label",
        dest="label_col",
        default="SYMP",
        help="Metadata label column (default: SYMP)."
    )

    parser.add_argument(
        "--k",
        type=positive_int,
        default=100000,
        help="Number of top features to retain (default: 100000)."
    )

    # Retained so existing HPC commands do not fail.
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help=(
            "Accepted for backwards compatibility. "
            "Not used by the streaming implementation."
        )
    )

    # Retained so existing HPC commands do not fail.
    parser.add_argument(
        "--block-cols",
        type=positive_int,
        default=50000,
        help=(
            "Accepted for backwards compatibility. "
            "Not used by the streaming implementation."
        )
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(output_dir: str, name: str) -> str:

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    jobid = (
        os.environ.get("LSB_JOBID")
        or os.environ.get("SLURM_JOB_ID")
        or "local"
    )

    log_file = os.path.join(
        output_dir,
        f"{name}_chisq_stream_{timestamp}_{jobid}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(
                log_file,
                mode="w"
            ),
            logging.StreamHandler(
                sys.stdout
            ),
        ],
        force=True,
    )

    return log_file


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def read_meta(
    meta_file: str,
    label_col: str
) -> pd.DataFrame:

    df = pd.read_csv(
        meta_file,
        sep="\t",
        header=0,
        dtype=str,
        index_col=0,
    )

    df.index = (
        df.index.astype(str)
    )

    if not df.index.is_unique:

        raise ValueError(
            "Duplicate sample IDs found in metadata file."
        )

    if label_col not in df.columns:

        raise ValueError(
            f"Missing required column '{label_col}' "
            "in metadata file."
        )

    return df[
        [label_col]
    ]


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def read_raw_header(
    feature_file: str
) -> Tuple[
    str,
    List[str],
    int,
    Optional[np.ndarray],
]:
    """
    Read the feature header directly as raw bytes.

    Returns
    -------
    id_col_name
        Name of the first/sample-ID column.

    feature_names
        Feature names retained for Chi², after dropping names beginning
        with "Unnamed" to preserve previous behavior.

    raw_feature_count
        Number of numeric feature fields in every source row before dropping
        any "Unnamed" columns.

    keep_positions
        None if every raw feature is retained.

        Otherwise a zero-based integer array giving the retained feature
        positions within the numeric portion of each row.
    """

    logging.info(
        "Reading raw feature header"
    )

    with open(
        feature_file,
        "rb",
        buffering=16 * 1024 * 1024,
    ) as handle:

        header = handle.readline()

    if not header:

        raise ValueError(
            "Feature matrix is empty."
        )

    header = header.rstrip(
        b"\r\n"
    )

    raw_fields = header.split(
        b"\t"
    )

    if len(raw_fields) < 2:

        raise ValueError(
            "Features file must contain at least "
            "two columns (ID + >=1 feature)."
        )

    id_col_name = raw_fields[0].decode(
        "utf-8",
        errors="strict"
    )

    raw_feature_count = (
        len(raw_fields) - 1
    )

    feature_names = []
    kept_positions_list = []

    dropped_unnamed = 0

    for raw_position, raw_name in enumerate(
        raw_fields[1:]
    ):

        name = raw_name.decode(
            "utf-8",
            errors="strict"
        )

        if str(name).startswith(
            "Unnamed"
        ):

            dropped_unnamed += 1
            continue

        feature_names.append(
            name
        )

        kept_positions_list.append(
            raw_position
        )

    if not feature_names:

        raise ValueError(
            "No usable feature columns found."
        )

    if dropped_unnamed == 0:

        keep_positions = None

    else:

        keep_positions = np.asarray(
            kept_positions_list,
            dtype=np.int64,
        )

    logging.info(
        "Raw feature count: %d",
        raw_feature_count
    )

    logging.info(
        "Feature count after header cleanup: %d",
        len(feature_names)
    )

    if dropped_unnamed:

        logging.info(
            "Dropped %d feature columns beginning with 'Unnamed'",
            dropped_unnamed
        )

    # Explicitly release millions of raw byte fields.
    del raw_fields
    del kept_positions_list

    return (
        id_col_name,
        feature_names,
        raw_feature_count,
        keep_positions,
    )


# ---------------------------------------------------------------------------
# Memory monitor
# ---------------------------------------------------------------------------

def monitor_memory(
    interval: int = 5
):
    """
    Report RSS of the current Python process.
    """

    proc = psutil.Process(
        os.getpid()
    )

    while True:

        mem = (
            proc.memory_info().rss
            / (1024 ** 3)
        )

        sys.stdout.write(
            f"\r[Monitor] Memory usage: "
            f"{mem:.2f} GB "
        )

        sys.stdout.flush()

        time.sleep(
            interval
        )


# ---------------------------------------------------------------------------
# Pass A: stream source matrix and accumulate Chi² sufficient statistics
# ---------------------------------------------------------------------------

def stream_class_feature_counts(
    feature_file: str,
    meta_df: pd.DataFrame,
    label_col: str,
    raw_feature_count: int,
    retained_feature_count: int,
    keep_positions: Optional[np.ndarray],
):
    """
    Scan the source feature matrix exactly once.

    For each sample shared with metadata:
      - read label
      - parse numeric row with np.fromstring()
      - binarize (>0)
      - add presence vector to that class's feature counts

    Returns
    -------
    observed
        uint32 array of shape:
            (n_active_classes, n_retained_features)

    class_sample_counts
        Number of retained samples in each active class.

    active_classes
        Class labels used in the calculation.

    row_index
        Shared sample IDs in original feature-file row order.
    """

    meta_labels = meta_df[
        label_col
    ]

    # Use metadata labels to establish stable class slots.
    nonmissing_labels = [
        str(value)
        for value in meta_labels.tolist()
        if not pd.isna(value)
    ]

    all_classes = sorted(
        set(nonmissing_labels)
    )

    if not all_classes:

        raise ValueError(
            "Metadata contains no non-missing labels."
        )

    class_to_index = {
        label: i
        for i, label
        in enumerate(all_classes)
    }

    observed = np.zeros(
        (
            len(all_classes),
            retained_feature_count
        ),
        dtype=np.uint32,
    )

    class_sample_counts = np.zeros(
        len(all_classes),
        dtype=np.int64,
    )

    meta_set = set(
        meta_df.index
    )

    seen_ids = set()

    row_index = []

    source_rows = 0

    logging.info(
        "Pass A: streaming feature matrix once for Chi² counts"
    )

    with open(
        feature_file,
        "rb",
        buffering=16 * 1024 * 1024,
    ) as source:

        # Skip raw header; it has already been parsed once.
        header = source.readline()

        if not header:

            raise ValueError(
                "Feature matrix is empty."
            )

        for line in tqdm(
            source,
            desc="Chi² streaming pass",
            unit="rows",
        ):

            source_rows += 1

            line = line.rstrip(
                b"\r\n"
            )

            if not line:

                continue

            sample_bytes, sep, numeric_bytes = (
                line.partition(
                    b"\t"
                )
            )

            if not sep:

                raise ValueError(
                    f"Source row {source_rows} "
                    "contains no tab delimiter."
                )

            sample_id = sample_bytes.decode(
                "utf-8",
                errors="strict"
            )

            if sample_id in seen_ids:

                raise ValueError(
                    "Duplicate sample ID found "
                    f"in feature matrix: {sample_id}"
                )

            seen_ids.add(
                sample_id
            )

            # Skip non-metadata samples before parsing millions of values.
            if sample_id not in meta_set:

                continue

            label = meta_df.at[
                sample_id,
                label_col
            ]

            if pd.isna(label):

                raise ValueError(
                    "Missing label found among samples "
                    f"used for Chi²: {sample_id}"
                )

            label = str(
                label
            )

            class_idx = class_to_index[
                label
            ]

            values = np.fromstring(
                numeric_bytes,
                sep="\t",
                dtype=np.int16,
            )

            if values.size != raw_feature_count:

                raise ValueError(
                    f"Row '{sample_id}' parsed into "
                    f"{values.size} feature values, "
                    f"but {raw_feature_count} were expected. "
                    "The row may be malformed or contain a value "
                    "numpy.fromstring() could not parse."
                )

            if keep_positions is None:

                present = (
                    values > 0
                )

            else:

                present = (
                    values[
                        keep_positions
                    ] > 0
                )

            observed[
                class_idx,
                :
            ] += present

            class_sample_counts[
                class_idx
            ] += 1

            row_index.append(
                sample_id
            )

            del present
            del values

    if not row_index:

        raise ValueError(
            "No overlapping sample IDs between "
            "metadata and feature matrix."
        )

    # Remove metadata classes for which no source sample was present.
    active_mask = (
        class_sample_counts > 0
    )

    observed = observed[
        active_mask,
        :
    ]

    class_sample_counts = (
        class_sample_counts[
            active_mask
        ]
    )

    active_classes = [
        label
        for label, active
        in zip(
            all_classes,
            active_mask
        )
        if active
    ]

    if len(active_classes) < 2:

        raise ValueError(
            "Chi² feature selection requires "
            "at least two classes among samples "
            "shared by metadata and feature matrix."
        )

    logging.info(
        "Source rows scanned in Pass A: %d",
        source_rows
    )

    logging.info(
        "Samples shared with metadata: %d",
        len(row_index)
    )

    logging.info(
        "Active classes: %s",
        ", ".join(
            active_classes
        )
    )

    logging.info(
        "Samples per class: %s",
        ", ".join(
            f"{label}={count}"
            for label, count
            in zip(
                active_classes,
                class_sample_counts
            )
        )
    )

    return (
        observed,
        class_sample_counts,
        active_classes,
        row_index,
    )


# ---------------------------------------------------------------------------
# Exact sklearn-style Chi² calculation from accumulated counts
# ---------------------------------------------------------------------------

def calculate_chi2_from_counts(
    observed: np.ndarray,
    class_sample_counts: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Reproduce sklearn.feature_selection.chi2 from class-feature counts.

    sklearn computes:

        feature_count = X.sum(axis=0)
        class_prob    = class sizes / n_samples
        expected      = class_prob[:, None] * feature_count

        chi2 = sum((observed - expected)^2 / expected, axis=classes)

    p-values use a Chi² distribution with:
        df = n_classes - 1
    """

    n_classes = observed.shape[0]

    n_samples = int(
        class_sample_counts.sum()
    )

    logging.info(
        "Calculating Chi² statistics for %d features "
        "across %d classes",
        observed.shape[1],
        n_classes,
    )

    feature_count = (
        observed.sum(
            axis=0,
            dtype=np.uint64
        )
        .astype(
            np.float64,
            copy=False
        )
    )

    scores = np.zeros(
        observed.shape[1],
        dtype=np.float64,
    )

    # Do one class at a time to avoid creating a large
    # n_classes × n_features float64 expected matrix.
    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        for class_idx in range(
            n_classes
        ):

            class_prob = (
                class_sample_counts[
                    class_idx
                ]
                / n_samples
            )

            expected = (
                class_prob
                * feature_count
            )

            observed_class = (
                observed[
                    class_idx,
                    :
                ].astype(
                    np.float64,
                    copy=False
                )
            )

            contribution = (
                (
                    observed_class
                    - expected
                ) ** 2
            ) / expected

            scores += contribution

            del contribution
            del expected
            del observed_class

    p_values = special.chdtrc(
        n_classes - 1,
        scores
    )

    return (
        scores,
        p_values,
    )


# ---------------------------------------------------------------------------
# P-values + top-k
# ---------------------------------------------------------------------------

def write_pvalues_and_select_topk(
    out_pvals: str,
    feature_names: List[str],
    p_values: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> Tuple[
    List[str],
    List[int],
]:
    """
    Write p-values in original retained-feature order and reproduce the
    previous script's bounded-heap top-k selection behavior.

    Returns
    -------
    topk_names_in_order
        Selected names restored to original feature order.

    topk_feature_indices
        Zero-based positions within feature_names, also in original order.
    """

    topk_heap = []

    logging.info(
        "Writing p-values and selecting top %d features",
        k
    )

    with open(
        out_pvals,
        "w",
        buffering=16 * 1024 * 1024,
    ) as handle:

        handle.write(
            "feature\tp_value\n"
        )

        for feature, pval, score in tqdm(
            zip(
                feature_names,
                p_values,
                scores
            ),
            total=len(feature_names),
            desc="Writing p-values",
        ):

            handle.write(
                f"{feature}\t{pval}\n"
            )

            # Preserve previous behavior:
            # features with undefined Chi² are not eligible for top-k.
            if not np.isfinite(
                score
            ):

                continue

            if len(topk_heap) < k:

                heappush(
                    topk_heap,
                    (
                        score,
                        feature
                    )
                )

            elif score > topk_heap[0][0]:

                heappop(
                    topk_heap
                )

                heappush(
                    topk_heap,
                    (
                        score,
                        feature
                    )
                )

    topk_set = {
        feature
        for _, feature
        in topk_heap
    }

    logging.info(
        "Identifying selected features in original column order"
    )

    topk_feature_indices = [
        idx
        for idx, feature
        in enumerate(
            feature_names
        )
        if feature in topk_set
    ][:k]

    topk_names_in_order = [
        feature_names[idx]
        for idx
        in topk_feature_indices
    ]

    logging.info(
        "Top-k feature list ready: %d features",
        len(topk_names_in_order)
    )

    return (
        topk_names_in_order,
        topk_feature_indices,
    )


# ---------------------------------------------------------------------------
# Efficient row output helper
# ---------------------------------------------------------------------------

def binary_feature_row(
    selected_values: np.ndarray
) -> bytes:
    """
    Convert uint8 0/1 values to tab-separated ASCII bytes without creating
    one Python string per feature.

    Example:
        [0, 1, 1, 0]
    becomes:
        b"0\\t1\\t1\\t0"
    """

    n = selected_values.size

    if n == 0:

        return b""

    out = np.empty(
        2 * n - 1,
        dtype=np.uint8,
    )

    out[
        0::2
    ] = (
        selected_values
        + ord("0")
    )

    out[
        1::2
    ] = ord("\t")

    return out.tobytes()


# ---------------------------------------------------------------------------
# Pass B: stream source again and write selected matrix directly
# ---------------------------------------------------------------------------

def write_selected_matrix_streaming(
    feature_file: str,
    out_topk: str,
    id_col_name: str,
    topk_names: List[str],
    topk_feature_indices: List[int],
    keep_positions: Optional[np.ndarray],
    raw_feature_count: int,
    row_index: List[str],
):
    """
    Scan the source matrix exactly once more and write only selected features.

    No pandas matrix parsing.
    No memmap.
    No full selected matrix in RAM.
    """

    row_set = set(
        row_index
    )

    # Convert selected retained-feature indices to raw numeric positions.
    if keep_positions is None:

        selected_raw_positions = np.asarray(
            topk_feature_indices,
            dtype=np.int64,
        )

    else:

        selected_raw_positions = (
            keep_positions[
                np.asarray(
                    topk_feature_indices,
                    dtype=np.int64,
                )
            ]
        )

    out_dir = os.path.dirname(
        out_topk
    ) or "."

    tmp_path = (
        out_topk
        + ".tmp"
    )

    written_rows = 0

    logging.info(
        "Pass B: streaming source matrix once to write "
        "%d selected features",
        len(topk_names)
    )

    try:

        with open(
            feature_file,
            "rb",
            buffering=16 * 1024 * 1024,
        ) as source, open(
            tmp_path,
            "wb",
            buffering=16 * 1024 * 1024,
        ) as out:

            source.readline()

            # Output header.
            out.write(
                id_col_name.encode(
                    "utf-8"
                )
            )

            if topk_names:

                out.write(
                    b"\t"
                )

                out.write(
                    "\t".join(
                        topk_names
                    ).encode(
                        "utf-8"
                    )
                )

            out.write(
                b"\n"
            )

            for line in tqdm(
                source,
                desc="Top-k streaming pass",
                unit="rows",
            ):

                line = line.rstrip(
                    b"\r\n"
                )

                if not line:

                    continue

                sample_bytes, sep, numeric_bytes = (
                    line.partition(
                        b"\t"
                    )
                )

                if not sep:

                    raise ValueError(
                        "Feature matrix contains a row "
                        "without a tab delimiter."
                    )

                sample_id = sample_bytes.decode(
                    "utf-8",
                    errors="strict"
                )

                # Skip before numeric parsing whenever possible.
                if sample_id not in row_set:

                    continue

                values = np.fromstring(
                    numeric_bytes,
                    sep="\t",
                    dtype=np.int16,
                )

                if values.size != raw_feature_count:

                    raise ValueError(
                        f"Row '{sample_id}' parsed into "
                        f"{values.size} feature values, "
                        f"but {raw_feature_count} were expected."
                    )

                selected_values = (
                    values[
                        selected_raw_positions
                    ] > 0
                ).astype(
                    np.uint8,
                    copy=False,
                )

                out.write(
                    sample_bytes
                )

                if selected_values.size:

                    out.write(
                        b"\t"
                    )

                    out.write(
                        binary_feature_row(
                            selected_values
                        )
                    )

                out.write(
                    b"\n"
                )

                written_rows += 1

                del selected_values
                del values

        if written_rows != len(row_index):

            raise ValueError(
                "Pass B retained a different number of samples "
                f"than Pass A: {written_rows} vs {len(row_index)}."
            )

        # Only expose final output after successful completion.
        os.replace(
            tmp_path,
            out_topk
        )

    except Exception:

        try:

            os.remove(
                tmp_path
            )

        except FileNotFoundError:

            pass

        raise

    logging.info(
        "Top-k output rows written: %d",
        written_rows
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    args = get_opts()

    log_file = setup_logging(
        args.output_dir,
        args.name
    )

    logging.info(
        "Writing Chi² log to %s",
        log_file
    )

    logging.info(
        "Streaming implementation active: "
        "--n_jobs and --block-cols are accepted but not used."
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    meta_df = read_meta(
        args.meta,
        args.label_col
    )

    logging.info(
        "Metadata rows: %d",
        len(meta_df)
    )

    # ------------------------------------------------------------------
    # Raw header only — no pandas parsing of ultra-wide matrix.
    # ------------------------------------------------------------------

    (
        id_col_name,
        feature_names,
        raw_feature_count,
        keep_positions,
    ) = read_raw_header(
        args.features1
    )

    retained_feature_count = (
        len(feature_names)
    )

    # Start lightweight memory monitor.
    threading.Thread(
        target=monitor_memory,
        daemon=True
    ).start()

    # ------------------------------------------------------------------
    # PASS A:
    # one source scan for all Chi² sufficient statistics.
    # ------------------------------------------------------------------

    (
        observed,
        class_sample_counts,
        active_classes,
        row_index,
    ) = stream_class_feature_counts(
        feature_file=args.features1,
        meta_df=meta_df,
        label_col=args.label_col,
        raw_feature_count=raw_feature_count,
        retained_feature_count=retained_feature_count,
        keep_positions=keep_positions,
    )

    # ------------------------------------------------------------------
    # Calculate sklearn-equivalent scores from accumulated counts.
    # ------------------------------------------------------------------

    (
        scores,
        p_values
    ) = calculate_chi2_from_counts(
        observed=observed,
        class_sample_counts=class_sample_counts,
    )

    # Counts are no longer needed.
    del observed

    # ------------------------------------------------------------------
    # Write p-values and select top-k.
    # ------------------------------------------------------------------

    out_pvals = os.path.join(
        args.output_dir,
        f"{args.name}_pvalues.tsv"
    )

    (
        topk_names,
        topk_feature_indices,
    ) = write_pvalues_and_select_topk(
        out_pvals=out_pvals,
        feature_names=feature_names,
        p_values=p_values,
        scores=scores,
        k=args.k,
    )

    # Scores/p-values are no longer needed.
    del scores
    del p_values

    # ------------------------------------------------------------------
    # PASS B:
    # one source scan to produce final selected matrix.
    # ------------------------------------------------------------------

    out_topk = os.path.join(
        args.output_dir,
        f"{args.name}_top{args.k}_features.tsv"
    )

    if topk_names:

        write_selected_matrix_streaming(
            feature_file=args.features1,
            out_topk=out_topk,
            id_col_name=id_col_name,
            topk_names=topk_names,
            topk_feature_indices=topk_feature_indices,
            keep_positions=keep_positions,
            raw_feature_count=raw_feature_count,
            row_index=row_index,
        )

    else:

        with open(
            out_topk,
            "w"
        ) as handle:

            handle.write(
                f"{id_col_name}\n"
            )

    # ------------------------------------------------------------------
    # Final log
    # ------------------------------------------------------------------

    logging.info(
        "Original raw feature count: %d",
        raw_feature_count
    )

    logging.info(
        "Feature count after header cleanup: %d",
        retained_feature_count
    )

    logging.info(
        "Samples used for Chi²: %d",
        len(row_index)
    )

    logging.info(
        "Reduced feature count top %d: %d",
        args.k,
        len(topk_names)
    )

    logging.info(
        "Saved top %d features to: %s",
        args.k,
        out_topk
    )

    logging.info(
        "Saved p-values to: %s",
        out_pvals
    )


if __name__ == "__main__":
    main()

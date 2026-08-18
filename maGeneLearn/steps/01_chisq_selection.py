#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_chisq_selection.py
=====================

Purpose:
    Chi-squared feature selection on very large genomic matrices in a memory-efficient,
    block-wise way (no per-row dense loops, no full densification).

Inputs:
    --meta       Path to metadata file (TSV containing sample identifier and label columns).
    --features1  Path to feature matrix (TSV, rows = isolates, cols = features, first col = sample IDs).
    --output_dir Directory to write output files.
    --name       Base name for output files (no extension).

Parameters:
    --label      Column name in metadata for labels (default: 'SYMP').
    --k          Number of top features to select (default: 100000).

Outputs (same filenames/formats as before):
    <name>_top{k}_features.tsv    Top k features (isolates × selected features; dense TSV).
    <name>_pvalues.tsv            2 columns: feature, p_value (all tested features).

"""

import os
import argparse
import tempfile
import numpy as np
import pandas as pd
from heapq import heappush, heappop
from typing import List, Tuple, Iterable, Dict
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil, threading, time, sys
from datetime import datetime
import logging

# Internal tunables (no new CLI flags)
_ROW_CHUNK  = 1000   # rows per write chunk when emitting TSVs


# ---------------- CLI ----------------

def get_opts():
    parser = argparse.ArgumentParser(
        description="Chi2 feature selection from genomic data (block-wise, memory-efficient)."
    )
    parser.add_argument('--meta', required=True, help='Path to metadata file (TSV with ID/label columns)')
    parser.add_argument('--features1', required=True, help='Path to feature matrix (TSV; first col = sample IDs)')
    parser.add_argument('--output_dir', required=True, help='Directory to write output files')
    parser.add_argument('--name', required=True, help='Base name for output files (no extension)')
    parser.add_argument('--label', dest='label_col', default='SYMP',
                        help="Metadata column name for labels (default: 'SYMP')")
    parser.add_argument('--k', type=int, default=100000,
                        help='Number of top features to select (default: 100000)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs to run (default: -1, use all available CPUs)')
    parser.add_argument(
        "--block-cols",
        type=int,
        default=50000,
        help="Number of feature columns read per block"
    )
    return parser.parse_args()


# ------------- Helpers -------------

def setup_logging(output_dir: str, name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobid = os.environ.get("SLURM_JOB_ID", "local")

    log_file = os.path.join(
        output_dir,
        f"{name}_chisq_{timestamp}_{jobid}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return log_file

def read_meta(meta_file: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(
        meta_file,
        sep="\t",
        header=0,
        dtype=str,
        index_col=0
    )

    if not df.index.is_unique:
        raise ValueError("Duplicate sample IDs found in metadata file.")

    if label_col not in df.columns:
        raise ValueError(
            f"Missing required column '{label_col}' in metadata file"
        )

    return df[[label_col]]

def scan_header_and_rows(feature_file: str) -> Tuple[str, List[str], List[str]]:
    """
    Read only the header to get column names and the sample-ID column name.
    Also read the first column (sample IDs) to get row order.
    No length-based filtering; we only drop accidental 'Unnamed' columns.
    """
    # Header
    header_df = pd.read_csv(feature_file, sep='\t', nrows=0)
    columns = list(header_df.columns)
    if len(columns) < 2:
        raise ValueError("Features file must have at least two columns (ID + ≥1 feature).")
    id_col_name = columns[0]
    # Drop accidental unnamed columns to match typical pandas behavior in older script
    all_feature_cols = [c for c in columns[1:] if not str(c).startswith("Unnamed")]
    keep_cols = all_feature_cols

    # Row order from file
    ids = pd.read_csv(
        feature_file,
        sep="\t",
        usecols=[id_col_name],
        dtype=str
    )[id_col_name].tolist()

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate sample IDs found in feature matrix."
        )
    return id_col_name, keep_cols, ids

def make_row_index(file_ids: List[str], meta_index: pd.Index) -> List[str]:
    meta_set = set(meta_index)
    return [rid for rid in file_ids if rid in meta_set]

def iter_column_blocks(cols: List[str], block_size: int) -> Iterable[List[str]]:
    for i in range(0, len(cols), block_size):
        yield cols[i:i+block_size]

def load_block_as_csr(feature_file: str, id_col_name: str, block_cols: List[str], row_index: List[str]) -> sparse.csr_matrix:
    usecols = [id_col_name] + block_cols
    df = pd.read_csv(
        feature_file, sep='\t', usecols=usecols,
        dtype={id_col_name: str, **{c: 'Int16' for c in block_cols}}
    )
    df = df.set_index(id_col_name).reindex(row_index).fillna(0)
    arr = (df.to_numpy(copy=False) > 0).astype(np.uint8, copy=False)
    return sparse.csr_matrix(arr, dtype=np.uint8)

def monitor_memory(interval=5):
    """Print memory usage every `interval` seconds in the background."""
    proc = psutil.Process(os.getpid())
    while True:
        mem = proc.memory_info().rss / (1024**3)  # GB
        sys.stdout.write(f"\r[Monitor] Memory usage: {mem:.2f} GB ")
        sys.stdout.flush()
        time.sleep(interval)

def chi2_block(block, feature_file, id_col_name, row_index, y):
    """Run chi2 on one block of features and return results."""
    X_block = load_block_as_csr(feature_file, id_col_name, block, row_index)
    chi2_scores, p_values = chi2(X_block, y)
    return list(zip(block, p_values, chi2_scores))

def build_memmap(path: str, shape: Tuple[int, int]) -> np.memmap:
    return np.memmap(
        path,
        dtype="uint8",
        mode="w+",
        shape=shape
    )

def fill_memmap_columns_from_blocks(
    mm: np.memmap,
    feature_file: str,
    id_col_name: str,
    row_index: List[str],
    selected_cols_in_order: List[str],
    block_size: int,
):
    total_blocks = (
        len(selected_cols_in_order) + block_size - 1
    ) // block_size

    offset = 0

    for block in tqdm(
        iter_column_blocks(selected_cols_in_order, block_size),
        total=total_blocks,
        desc="Building matrix"
    ):
        df = pd.read_csv(
            feature_file,
            sep="\t",
            usecols=[id_col_name] + block,
            dtype={id_col_name: str, **{c: "Int16" for c in block}}
        ).set_index(id_col_name)

        # Ensure both rows and feature columns are in exactly
        # the expected order
        df = df.reindex(index=row_index, columns=block).fillna(0)

        arr = (df.to_numpy(copy=False) > 0).astype(
            np.uint8,
            copy=False
        )

        # Write the entire block to the memmap at once
        end = offset + len(block)
        mm[:, offset:end] = arr
        offset = end

    mm.flush()

def write_memmap_matrix_as_tsv(
    out_path: str,
    mm_path: str,
    shape: Tuple[int, int],
    row_ids: List[str],
    col_names: List[str],
    row_chunk: int = _ROW_CHUNK,
):
    mm = np.memmap(mm_path, dtype='uint8', mode='r', shape=shape)
    first = True
    with open(out_path, "w") as f:
        for start in range(0, shape[0], row_chunk):
            end = min(start + row_chunk, shape[0])
            chunk = np.asarray(mm[start:end, :])
            df = pd.DataFrame(chunk, index=row_ids[start:end], columns=col_names)
            df.to_csv(f, sep="\t", header=first, index=True, index_label=None, mode='a')
            first = False
    del mm


# --------------- MAIN ---------------

if __name__ == "__main__":
    args = get_opts()
    log_file = setup_logging(args.output_dir, args.name)
    logging.info("Writing Chi² log to %s", log_file)

    # Metadata & row order
    meta_df = read_meta(args.meta,args.label_col)
    id_col_name, keep_cols, file_row_ids = scan_header_and_rows(args.features1)
    row_index = make_row_index(file_row_ids, meta_df.index)
    if len(row_index) == 0:
        raise ValueError("No overlapping sample IDs between metadata and features file.")

    # Labels aligned to row order
    labels = meta_df.loc[row_index, args.label_col]

    if labels.isna().any():
        raise ValueError("Missing labels found among samples used for Chi².")

    y = LabelEncoder().fit_transform(labels)

    if np.unique(y).size < 2:
        raise ValueError(
            "Chi² feature selection requires at least two classes."
        )

    # -------- Pass A: chi² scoring in blocks; parallel; progress bar; memory monitor --------
    logging.info("Scoring Chi² in blocks over %d features; rows=%d", len(keep_cols), len(row_index))
    out_pvals = os.path.join(args.output_dir, f"{args.name}_pvalues.tsv")

    # Start memory monitor in background
    threading.Thread(target=monitor_memory, daemon=True).start()

    # Sequential block processing:
    # read one block -> run Chi² -> move to next block
    topk_heap = []

    total_blocks = (
                           len(keep_cols) + args.block_cols - 1
                   ) // args.block_cols

    parallel_results = Parallel(
        n_jobs=args.n_jobs,
        return_as="generator",
    )(
        delayed(chi2_block)(
            block,
            args.features1,
            id_col_name,
            row_index,
            y
        )
        for block in iter_column_blocks(
            keep_cols,
            args.block_cols
        )
    )

    with open(out_pvals, "w") as w:
        w.write("feature\tp_value\n")

        for block_results in tqdm(
                parallel_results,
                total=total_blocks,
                desc="Chi² blocks"
        ):
            for feat, pval, score in block_results:
                w.write(f"{feat}\t{pval}\n")

                # Skip features for which Chi² is undefined
                if not np.isfinite(score):
                    continue

                if len(topk_heap) < args.k:
                    heappush(topk_heap, (score, feat))

                elif score > topk_heap[0][0]:
                    heappop(topk_heap)
                    heappush(topk_heap, (score, feat))

    topk_set = {feat for _, feat in topk_heap}
    logging.info("Identifying top %d features in original column order", args.k)

    topk_in_order = [
        c for c in keep_cols
        if c in topk_set
    ][:args.k]

    logging.info("Top-k feature list ready: %d features", len(topk_in_order))

    # ---------------- Pass B: build & write Top-K matrix ----------------
    out_topk = os.path.join(args.output_dir, f"{args.name}_top{args.k}_features.tsv")
    if len(topk_in_order) > 0:
        with tempfile.NamedTemporaryFile(delete=False, dir=args.output_dir, prefix=f"{args.name}_topk_", suffix=".mm") as tmpf:
            mm_path = tmpf.name
        mm = build_memmap(mm_path, shape=(len(row_index), len(topk_in_order)))
        fill_memmap_columns_from_blocks(mm, args.features1, id_col_name, row_index, topk_in_order, args.block_cols)
        del mm
        write_memmap_matrix_as_tsv(out_topk, mm_path, (len(row_index), len(topk_in_order)), row_index, topk_in_order, row_chunk=_ROW_CHUNK)
        try: os.remove(mm_path)
        except Exception: pass
    else:
        # Write an empty header-only file (consistent with prior behavior when selection yields 0 cols)
        with open(out_topk, "w") as f:
            f.write("\t\n")

    # Logs
    logging.info("Original feature count after header cleanup: %d", len(keep_cols))
    logging.info(
        "Reduced feature count top %d: %d",
        args.k,
        len(topk_in_order)
    )
    logging.info("Saved top %d features to: %s", args.k, out_topk)
    logging.info("Saved p-values to: %s", out_pvals)


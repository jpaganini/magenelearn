import argparse
import logging
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

"""
00_split_dataset.py

Splits a metadata TSV into train/test sets stratified by outcome and grouped by clusters within each lineage.

Inputs (CLI args):
  --meta-file / -m      Path to input metadata TSV. Must contain columns:
                          * sample ID (specified via --id-col)
                          * lineage grouping (via --lineage-col)
                          * stratification outcome (via --outcome-col)
                          * clustering/group variable (via --group-col)
  --out-dir / -o        Output directory for result TSVs.
  --name / -n           Base filename; '_train.tsv' and '_test.tsv' appended.
  --lineage-col / -l    Column name for lineage grouping (default: LINEAGE).
  --group-col / -g      Column name for grouping clusters (default: t5).
  --id-col / -i         Column name for sample identifier (default: SRA).
  --outcome-col / -c    Column name for stratification outcome/label (default: SYMP).
  --n-splits / -k       Number of folds for StratifiedGroupKFold (default: 5).
  --seed / -s           Random seed for fold selection (default: 42).
  --print-metrics       If set, prints value counts for lineage, outcome, and group in train/test sets.

Outputs:
  <out-dir>/<name>_train.tsv  Train set (TSV).
  <out-dir>/<name>_test.tsv   Test set (TSV).
  
Usage Example:
python 00_split_dataset.py \
--meta-file data/metadata.tsv \
--out-dir results/ \
--name study1 \
--id-col SampleID \
--lineage-col Clade \
--group-col Batch \
--outcome-col Status \
--n-splits 5 \
--seed 123 \
--print-metrics
"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split metadata into train/test sets by lineage with stratified group k-fold."
    )
    parser.add_argument(
        "--meta-file", "-m",
        type=Path,
        required=True,
        help="Path to metadata TSV file."
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=Path,
        required=True,
        help="Directory to write train/test output files."
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        required=True,
        help="Base name for output files; '_train.tsv' and '_test.tsv' will be appended."
    )
    parser.add_argument(
        "--lineage-col", "-l",
        type=str,
        default="LINEAGE",
        help="Column name to use for lineage grouping (default: LINEAGE)."
    )
    parser.add_argument(
        "--group-col", "-g",
        type=str,
        default="t5",
        help="Column name to use as grouping variable for StratifiedGroupKFold (default: t5)."
    )
    parser.add_argument(
        "--id-col", "-i",
        type=str,
        default="SRA",
        help="Column name to use as the sample ID/index (default: SRA)."
    )
    parser.add_argument(
        "--outcome-col", "-c",
        type=str,
        default="SYMP",
        help="Column name to use as the outcome/label (default: SYMP)."
    )
    parser.add_argument(
        "--n-splits", "-k",
        type=int,
        default=5,
        help="Number of splits for StratifiedGroupKFold (default: 5)."
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for selecting one fold (default: 42)."
    )
    parser.add_argument(
        "--print-metrics",
        action="store_true",
        help="Print value counts for lineage, outcome, and group in train/test sets."
    )
    return parser.parse_args()


def split_by_lineage(
    metadata: pd.DataFrame,
    lineage_col: str,
    group_col: str,
    outcome_col: str,
    n_splits: int,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform stratified-group k-fold splitting within each lineage.

    Args:
        metadata: DataFrame indexed by sample ID, must include lineage_col, group_col, outcome_col.
        lineage_col: column name for lineages.
        group_col: column name for grouping clusters.
        outcome_col: column name for labels.
        n_splits: number of folds for StratifiedGroupKFold.
        seed: random seed to pick one of the folds.

    Returns:
        final_train, final_test: concatenated DataFrames.
    """
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    #rng = random.Random(seed)
    skf = StratifiedGroupKFold(n_splits=n_splits)

    for lineage, group_df in metadata.groupby(lineage_col):
        if group_df.shape[0] < n_splits:
            logging.warning(
                f"Skipping lineage '{lineage}' with {group_df.shape[0]} samples (< {n_splits} splits)."
            )
            continue

        labels = group_df[outcome_col].values
        clusters = group_df[group_col].values

        lineage_rng = random.Random(seed)
        splits = list(skf.split(group_df, labels, groups=clusters))
        train_idx, test_idx = lineage_rng.choice(splits)

        train_parts.append(group_df.iloc[train_idx])
        test_parts.append(group_df.iloc[test_idx])

    final_train = pd.concat(train_parts, ignore_index=False)
    final_test = pd.concat(test_parts, ignore_index=False)
    return final_train, final_test


def print_value_counts(
    df: pd.DataFrame,
    columns: List[str],
    label: str
) -> None:
    """
    Print value counts for specified columns in a DataFrame.
    """
    print(f"--- {label} Metrics ---")
    for col in columns:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False)
            print(f"{col}:\n{counts}\n")
        else:
            print(f"Column '{col}' not found in DataFrame.\n")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ensure output directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # load metadata
    metadata = pd.read_csv(
        args.meta_file,
        sep="\t",
        header=0,
        index_col=None
    )
    # set sample ID index
    if args.id_col not in metadata.columns:
        logging.error(f"ID column not found: {args.id_col}")
        return
    metadata = metadata.set_index(args.id_col)

    # check lineage, group, outcome columns
    for col in (args.lineage_col, args.group_col, args.outcome_col):
        if col not in metadata.columns:
            logging.error(f"Required column not found: {col}")
            return

    # split dataset
    train_df, test_df = split_by_lineage(
        metadata,
        lineage_col=args.lineage_col,
        group_col=args.group_col,
        outcome_col=args.outcome_col,
        n_splits=args.n_splits,
        seed=args.seed
    )

    # print metrics if requested
    if args.print_metrics:
        cols = [args.lineage_col, args.outcome_col, args.group_col]
        print_value_counts(train_df, cols, label="Train Set")
        print_value_counts(test_df, cols, label="Test Set")

    # write outputs
    train_path = args.out_dir / f"{args.name}_train.tsv"
    test_path = args.out_dir / f"{args.name}_test.tsv"
    train_df.to_csv(train_path, sep="\t")
    test_df.to_csv(test_path, sep="\t")

    logging.info(
        f"Train/test split complete: {len(train_df)} train -> {train_path}; "
        f"{len(test_df)} test -> {test_path}"
    )


if __name__ == "__main__":
    main()

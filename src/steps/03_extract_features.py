import os
import sys
import argparse
import pandas as pd
from pathlib import Path

"""
03_extract_features.py

Extracts a subset of user-selected features from a full feature matrix and merges with labels and group IDs.

Inputs (CLI args):
  --muvr_file     Path to MUVR-selected feature file (e.g. *_muvr_RFC_min.tsv).
                   Must include:
                     * sample ID as index
                     * one column with the chosen label (specified via --label)
                     * remaining columns are selected feature names.
  --chisq_file    Path to full feature matrix TSV (e.g. full_chisq_matrix.tsv).
                   Must include:
                     * sample ID as first column (will be used as index)
                     * all feature columns, from which selected ones will be extracted.
  --train_metadata   Path to metadata TSV for training split (index, label, group columns).
  --test_metadata    Path to metadata TSV for testing split (same columns as train_metadata).
  --label         Name of the label column in the MUVR file to include in output.
  --group_column  Name of the grouping column in the metadata file.
  --output_dir    Directory where the extracted feature matrix will be written.
                   Parent directories will be created if needed.
  --name          Base name (without extension) for the output file; “.tsv” will be appended.

Outputs:
  Two TSV files at <output_dir>/<name>_train.tsv and <output_dir>/<name>_test.tsv, each containing, for each sample:
    * extracted features (columns matching those in the MUVR file minus the label)
    * the original label column with its original name
    * the original group column with its original name

Usage Example:
  python 03_extract_features.py \
    --muvr_file results/muvr_RFC_min.tsv \
    --chisq_file data/full_chisq_matrix.tsv \
    --train_metadata data/train_metadata.tsv \
    --test_metadata data/test_metadata.tsv \
    --label SYMP \
    --group_column cohort \
    --output_dir results \
    --name features_min_merged
"""

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract MUVR-selected features from a full feature matrix for all samples (train/test)."
    )
    parser.add_argument('--muvr_file', type=str, required=True,
                        help='Path to MUVR-selected feature file (e.g. *_muvr_RFC_min.tsv)')
    parser.add_argument('--chisq_file', type=str, required=True,
                        help='Path to full feature matrix (e.g. full_chisq_matrix.tsv)')
    parser.add_argument('--train_metadata', type=str, required=True,
                        help='Path to metadata TSV for training split')
    parser.add_argument('--test_metadata', type=str, required=True,
                        help='Path to metadata TSV for testing split')
    parser.add_argument('--label', type=str, required=True,
                        help='Name of the label column to include in output')
    parser.add_argument('--group_column', type=str, required=True,
                        help = 'Name of the grouping column in metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                        help = 'Directory where the extracted feature matrix will be written')
    parser.add_argument('--name', type=str, required=True,
                        help = 'Base name (without extension) for the output file')

    return parser.parse_args()


def load_selected(muvr_path, label_col):
    """
        Load the MUVR file, extract selected feature names and labels.
        Returns:
            features: list of feature column names
            labels: pd.Series indexed by sample ID
        """
    df = pd.read_csv(muvr_path, sep='\t', index_col=0)
    if label_col not in df.columns:
        raise SystemExit(f"Error: label '{label_col}' not found in MUVR file")
    # all other columns are selected features
    features = [c for c in df.columns if c != label_col]
    return features

def load_split_metadata(meta_path, label_col, group_col):
    """
    Load metadata split file, ensuring label and group columns exist.
    Returns:
        labels: pd.Series of label values indexed by sample ID
        groups: pd.Series of group IDs indexed by sample ID
    """
    meta = pd.read_csv(meta_path, sep='\t', index_col=0)
    missing = [col for col in (label_col, group_col) if col not in meta.columns]
    if missing:
        raise SystemExit(f"Error: columns {missing} not found in metadata file {meta_path}")
    return meta[label_col], meta[group_col]


def extract_features(chisq_file, selected_features):
    """Extract selected features from full chisq matrix."""
    #usecols = ['Unnamed: 0'] + selected_features  # 'Unnamed: 0' ensures the index column is loaded
    df_full = pd.read_csv(chisq_file, sep='\t', index_col=0)
    df = df_full[selected_features]
    return df


def process_split(meta_path, chisq_file, features, label_col, group_col, output_dir, suffix, base_name):
    """
    Load metadata, extract features, merge, and write one split with suffix.
    """
    labels, groups = load_split_metadata(meta_path, label_col, group_col)
    feats = extract_features(chisq_file, features)
    df = pd.concat([feats, labels, groups], axis=1, join='inner')

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{base_name}_{suffix}.tsv"
    final_path = outdir / name
    print(f"Saving {suffix} split to {final_path}")
    df.to_csv(final_path, sep='\t')

def main():
    args = parse_arguments()

    print(f"Loading selected features from {args.muvr_file}")
    features = load_selected(args.muvr_file, args.label)

    # Process train and test splits
    print("Processing train split...")
    process_split(
        args.train_metadata,
        args.chisq_file,
        features,
        args.label,
        args.group_column,
        args.output_dir,
        suffix="train",
        base_name=args.name
    )

    print("Processing test split...")
    process_split(
        args.test_metadata,
        args.chisq_file,
        features,
        args.label,
        args.group_column,
        args.output_dir,
        suffix="test",
        base_name=args.name
    )

    print("Done.")


if __name__ == "__main__":
    main()

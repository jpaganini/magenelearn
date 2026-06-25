import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

"""
03_extract_features.py

Extracts a subset of user-selected features from a full feature matrix and merges with labels and group IDs.

Inputs (CLI args):
  --feature_file     Path to MUVR-selected feature file (e.g. *_muvr_RFC_min.tsv).
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
    --feature_file results/muvr_RFC_min.tsv \
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
    parser.add_argument('--feature_file', type=str, required=True,
                        help='Path to MUVR-selected feature file (e.g. *_muvr_RFC_min.tsv)')
    parser.add_argument('--chisq_file', type=str, required=True,
                        help='Path to full feature matrix (e.g. full_chisq_matrix.tsv)')
    parser.add_argument('--train_metadata', type=str, required=False,
                        help='Path to metadata TSV for training split')
    parser.add_argument('--test_metadata', type=str, required=False,
                        help='Path to metadata TSV for testing split')
    parser.add_argument('--label', type=str, required=False,
                        help='Name of the label column to include in output')
    parser.add_argument('--group_column', type=str, required=False,
                        help = 'Name of the grouping column in metadata')
    parser.add_argument('--output_dir', type=str, required=True,
                        help = 'Directory where the extracted feature matrix will be written')
    parser.add_argument('--name', type=str, required=True,
                        help = 'Base name (without extension) for the output file')

    return parser.parse_args()



#helper functions
def setup_logging(output_dir: str, name: str) -> str:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = Path(output_dir) / f"{name}_extract_{timestamp}.log"

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


def _header_columns(path: Path) -> list[str]:
    """Read only the first line to get the complete header."""
    with path.open() as fh:
        return fh.readline().rstrip("\n").split("\t")


def extract_selected_columns(chisq_path: Path,
                             selected_cols: list[str]) -> pd.DataFrame:
    """
    Load only the selected k-mer columns and keep sample IDs as the index,
    even when the first header cell is blank.
    """
    header = _header_columns(chisq_path)          # list[str]
    pos_usecols = [0]                             # always keep column 0

    # map k-mer names → their integer positions
    name_to_pos = {c: i for i, c in enumerate(header)}
    missing = []

    for col in (c.strip() for c in selected_cols if c.strip()):
        if col in name_to_pos:
            pos_usecols.append(name_to_pos[col])
        else:
            missing.append(col)

    if len(pos_usecols) == 1:                     # only the ID column kept
        raise SystemExit("None of the selected features are present in "
                         f"{chisq_path.name}")

    if missing:
        logging.warning(
            "%d selected features not found in full matrix "
            "(showing first 5): %s",
            len(missing),
            missing[:5]
        )

    # ── read: homogeneous int list → pandas is happy
    df = pd.read_csv(
            chisq_path,
            sep="\t",
            usecols=pos_usecols,      # positions only
            header=0,                 # keep header row
            memory_map=True,
    )

    # column 0 is still present; make it the index
    df.set_index(df.columns[0], inplace=True)
    df.index = df.index.astype(str).str.strip()

    # 3️⃣  convert only the k-mer columns
    df.iloc[:, :] = df.iloc[:, :].astype("int8")

# We'll fill these missing columns with zeros later, so the model input shape is preserved.


    for col in missing:
        df[col] = 0

    df = df.reindex(columns=selected_cols, fill_value=0)

    return df

def load_selected(muvr_path, label_col):
    """
        Load the MUVR file, extract selected feature names and labels.
        Returns:
            features: list of feature column names
            labels: pd.Series indexed by sample ID
        """
    df = pd.read_csv(muvr_path, sep='\t', index_col=0)
    if label_col in df.columns:
        # all other columns are selected features
        features = [c for c in df.columns if c != label_col]
        return features

    return list(df.columns)

#def load_split_metadata(meta_path, label_col, group_col):
def load_split_metadata(meta_path, label_col, group_col=None):
    """
    Load metadata split file, ensuring label and group columns exist.
    Returns:
        labels: pd.Series of label values indexed by sample ID
        groups: pd.Series of group IDs indexed by sample ID
    """
    meta = pd.read_csv(meta_path, sep='\t', index_col=0)
    meta.index = meta.index.astype(str).str.strip()
    dupes = meta.index[meta.index.duplicated()]
    unique_dupes = dupes.unique()

    if len(dupes) > 0:
        logging.error(
            "%d duplicate rows involving %d sample IDs: %s",
            len(dupes),
            len(unique_dupes),
            list(unique_dupes[:10])
        )

        raise ValueError(
            f"{len(unique_dupes)} duplicate sample IDs found. "
            "Sample IDs must be unique."
        )


    if label_col is None:
        raise SystemExit("Error: --label is required when metadata is provided.")
    if label_col not in meta.columns:
        raise SystemExit(f"Error: label column '{label_col}' not found in metadata file {meta_path}")

    groups = None
    if group_col:
        if group_col in meta.columns:
            groups = meta[group_col]
        else:
            logging.warning(
                "Group column '%s' not found in %s; proceeding without groups.",
                group_col,
                meta_path
            )

    return meta[label_col], groups


def extract_features(chisq_file: str, selected_features: list[str]) -> pd.DataFrame:
    """Wrapper that calls the efficient column loader."""
    return extract_selected_columns(Path(chisq_file), selected_features)

# def extract_features(chisq_file, selected_features):
#     """Extract selected features from full chisq matrix."""
#     df_full = pd.read_csv(chisq_file, sep='\t', index_col=0)
#     df = df_full[selected_features]
#     return df


def process_split(meta_path, chisq_file, features, label_col, group_col, output_dir, suffix, base_name):
    """
    Load metadata, extract features, merge, and write one split with suffix.
    """
    labels, groups = load_split_metadata(meta_path, label_col, group_col)
    feats = extract_features(chisq_file, features)
#    df = pd.concat([feats, labels, groups], axis=1, join='inner')
    parts = [feats, labels]
    if groups is not None:
        parts.append(groups)

    logging.info("labels: %d", labels.shape[0])
    logging.info("features: %d", feats.shape[0])

    df = pd.concat(parts, axis=1, join='inner')

    logging.info("after merge: %d", df.shape[0])

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = f"{base_name}_{suffix}.tsv"
    final_path = outdir / name
    logging.info(
        "Saving %s split to %s",
        suffix,
        final_path
    )

    df.to_csv(final_path, sep='\t')

def main():
    args = parse_arguments()

    log_file = setup_logging(args.output_dir, args.name)

    logging.info("Writing extract log to %s", log_file)

    if not (args.train_metadata or args.test_metadata):
        logging.warning(
            "No metadata provided — extracting features only (predict-only mode)."
        )
    else:
        # Metadata provided => label must be provided and present in metadata
        if not args.label:
            raise SystemExit("Error: --label is required when --train_metadata/--test_metadata is provided.")
        # group_column may be None; that's OK

    logging.info(
        "Loading selected features from %s",
        args.feature_file
    )

    features = load_selected(args.feature_file, args.label)
    logging.info(
        "Loaded %d selected features",
        len(features)
    )


    # Process train or/and test splits
    if args.train_metadata:
        logging.info("Processing train split...")
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
    else:
        logging.info(
            "No train_metadata given – skipping train feature table construction."
        )

    if args.test_metadata:
        logging.info("Processing test split...")
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
    else:
        logging.info(
            "No test_metadata given – skipping hold-out feature table."
        )

    if not args.train_metadata and not args.test_metadata:
        # predict-only mode: extract and write features with no label/group
        feats = extract_features(args.chisq_file, features)
        outdir = Path(args.output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        final_path = outdir / f"{args.name}_test.tsv"
        feats.to_csv(final_path, sep="\t")
        logging.info(
            "Predict-only test table saved to %s",
            final_path
        )


    logging.info("Done.")


if __name__ == "__main__":
    main()

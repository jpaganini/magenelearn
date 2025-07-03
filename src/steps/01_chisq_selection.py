import os
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectKBest
import argparse

# ----------------------------------------------------------------------
# Script: 01_chisq_selection.py
#
# Purpose:
#     Perform Chi-squared feature selection on genomic k-mer data.
#
# Inputs:
#     --meta:       Path to metadata file (TSV containing sample identifier and label columns).
#     --features1:  Path to first feature matrix (TSV, k-mer counts).
#     --features2:  Path to second feature matrix (TSV, k-mer counts) [optional].
#
# Parameters:
#     --length_threshold:  Minimum k-mer string length to retain (default: 80).
#     --id:                Column name in metadata for sample identifiers (default: 'SRA').
#     --label:             Column name in metadata for target labels (default: 'SYMP').
#     --k:                 Number of top features to select (default: 100000).
#     --name:              Base name for output files (no extension).
#
# Outputs (written to --output_dir):
#     <name>_100k.tsv    Top 100,000 features selected by Chi-squared score.
#     <name>_pvalue.tsv  Features with p-value ≤ 0.05.
#     <name>_feature_pvalues.tsv    Table of all features and their p-values.
#
# Usage Example:
#     python 01_chisq_selection.py --meta meta.tsv \
#         --features1 feats1.tsv --features2 feats2.tsv \
#         --output_dir results/ --name study1
# ----------------------------------------------------------------------



########## FUNCTIONS ###############

def get_opts():
    parser = argparse.ArgumentParser(description="Chi2 feature selection from genomic data.")
    parser.add_argument('--meta', required=True, help='Path to metadata file (TSV with columns SRA, SYMP)')
    parser.add_argument('--features1', required=True, help='Path to first feature matrix (TSV)')
    parser.add_argument('--features2', required=False, help='Path to second feature matrix (TSV)')
    parser.add_argument('--output_dir', required=True, help='Directory to write output files')
    parser.add_argument('--name', required=True, help='Base name for output files (no extension)')
    parser.add_argument('--length_threshold', type=int, default=80, help='Minimum column name length to keep')
    parser.add_argument('--id', dest='id_col', default='SRA', help="Metadata column name for sample IDs (default: 'SRA')")
    parser.add_argument('--label', dest='label_col', default='SYMP', help="Metadata column name for labels (default: 'SYMP')")
    parser.add_argument('--k', type=int, default=100000, help='Number of top features to select (default: 100000)')
    return parser.parse_args()


def read_meta(meta_file, id_col, label_col):
    all_pd = pd.read_csv(meta_file, sep='\t', header=0)
    all_pd = all_pd[[id_col, label_col]]
    return all_pd.set_index(id_col)

def load_and_filter_features(feature_file, length_threshold):
    print(f"Loading {feature_file}")
    df = pd.read_csv(feature_file, sep='\t', header=0, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Binarize
    df[df > 0] = 1
    #df = df.T

    # Filter by k-mer length
    long_cols = [col for col in df.columns if len(col) >= length_threshold]
    df = df[long_cols]

    # Reduce memory usage
    df = df.astype('int8')
    return df

def merge_feature_sets(df1, df2):
    # Identify missing rows in each
    to_add1 = set(df1.index) - set(df2.index)
    to_add2 = set(df2.index) - set(df1.index)

    # Fill missing rows with 0s
    df1 = pd.concat([df1, pd.DataFrame(0, index=to_add2, columns=df1.columns).astype("int8")])
    df2 = pd.concat([df2, pd.DataFrame(0, index=to_add1, columns=df2.columns).astype("int8")])

    # Inner join on index
    return df1.join(df2, how='inner')

def perform_chi2_analysis(merged_df, label_col, k, output_dir, name):
    label_encoder = LabelEncoder()
    merged_df[label_col] = label_encoder.fit_transform(merged_df[label_col])

    X = merged_df.drop(label_col, axis=1)
    y = merged_df[label_col]

    chi_scores = chi2(X, y)

    # Select top 100k features
    selector = SelectKBest(chi2, k=k)
    X_kbest = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    print(f'Original feature count: {X.shape[1]}')
    print(f'Reduced feature count (top {k}): {X_kbest.shape[1]}')

    # Save top 100k features
    os.makedirs(output_dir, exist_ok=True)
    out_k = os.path.join(output_dir, f'{name}_top{k}_features.tsv')
    merged_df[selected_features].to_csv(out_k, sep='\t')

    df_pvalues = pd.DataFrame({
        'feature': X.columns,
        'p_value': chi_scores[1]
    })

    # Save full p-values table
    os.makedirs(output_dir, exist_ok=True)
    out_pvals = os.path.join(output_dir, f'{name}_pvalues.tsv')
    df_pvalues.to_csv(out_pvals, sep='	', index=False)


    # Save p-value filtered features (p ≤ 0.05)
    p_values = pd.Series(chi_scores[1], index=X.columns)
    good_cols = p_values[p_values <= 0.05].index
    out_pval = os.path.join(output_dir, f'{name}_pvalues_features.tsv')
    merged_df[good_cols].to_csv(out_pval, sep='\t')

    print(f'Saved top 100k features to: {out_k}')
    print(f'Saved p ≤ 0.05 features to: {out_pval}')

########### MAIN #############

if __name__ == "__main__":
    args = get_opts()

    # Load metadata with flexible column names
    meta_df = read_meta(args.meta, args.id_col, args.label_col)

    # Load and filter feature matrices

    features1 = load_and_filter_features(args.features1, args.length_threshold)
    if args.features2:
        features2 = load_and_filter_features(args.features2, args.length_threshold)
        combined = merge_feature_sets(features1, features2)
    else:
        print("No second feature file provided; using only features1")
        combined = features1.copy()

    #combined = combined.T

    print("Merging with metadata")
    merged_with_meta = pd.merge(combined, meta_df, left_index=True, right_index=True)

    perform_chi2_analysis(merged_with_meta, args.label_col, args.k, args.output_dir, args.name)




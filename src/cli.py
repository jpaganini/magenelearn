#!/usr/bin/env python3
"""
MageneLearn CLI (cli.py)

A Click-based wrapper to orchestrate the MaGene ML pipeline steps 00–05, with organized outputs.

Commands:
  Train    Execute dataset split, Chi², MUVR, feature extraction, training, and CV evaluation.
  Test     Run hold-out evaluation (no CV) on a trained model.

Required inputs:
  For Train mode:
    --meta-file      Path to metadata TSV (required)
    --features1      Path to first k-mer counts TSV (required)
    --name           Base filename prefix (required)
    --model          Model type: XGBC or RFC (required)
    [--chisq]        Flag to run Chi² feature selection (optional)
    [--muvr]         Flag to run MUVR feature selection (optional, requires --chisq)
    [--no-split]     Skip dataset splitting if already done (optional)
    [--features2]    Path to second k-mer counts TSV (optional)
    [--upsampling]   Upsampling method: none, smote, random (default: none)
    [--n-splits]     Number of CV folds (default: 5)
    [--output-dir]   Base output directory (default: timestamped YYMMDD_HHMM)

  For Test mode:
    --model-file     Path to trained .joblib model (required)
    --features       Path to features TSV (required)
    --name           Base filename prefix for outputs (required)
    [--label]        Label column name (default: outcome)
    [--group-column] Group column name (default: group)
    [--output-dir]   Base output directory (default: timestamped YYMMDD_HHMM)

Usage examples:
  maGeneLearn Train --meta-file meta.tsv --features1 kmers1.tsv --name exp --model RFC --chisq --muvr --upsampling smote --n-splits 10
  maGeneLearn Test --model-file exp_RFC_smote.joblib --features exp_test.tsv --name exp_test
"""
import click
import subprocess
import sys
import os
import glob
from datetime import datetime
from importlib import resources

# Determine where the numbered scripts are installed
try:
    STEPS_DIR = resources.files('maGeneLearn.steps')
except Exception:
    # fallback for Python <3.9 or edit-in-place
    STEPS_DIR = os.path.join(os.path.dirname(__file__), 'steps')


def run_cmd(cmd, cwd=None):
    """Run a command in subprocess, exiting on error."""
    click.echo(f"\n>>> Running: {' '.join(cmd)} (cwd={cwd})")
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode:
        sys.exit(res.returncode)


def make_step_dir(base_dir, idx, name):
    path = os.path.join(base_dir, f"{idx:02d}_{name}")
    os.makedirs(path, exist_ok=True)
    return path

@click.group()
def cli():
    """MaGeneLearn CLI entry point."""
    pass

@cli.command()
@click.option('--output-dir', default=None,
              help='Base output directory (default: timestamped YYMMDD_HHMM)')
@click.option('--chisq', is_flag=True, help='Run Chi² selection (01)')
@click.option('--muvr', is_flag=True, help='Run MUVR selection (02)')
@click.option('--no-split', 'no_split', is_flag=True, help='Skip splitting (00)')
@click.option('--meta-file', required=True, help='Metadata TSV')
@click.option('--features1', required=True, help='K-mer counts TSV #1')
@click.option('--features2', default=None, help='K-mer counts TSV #2 (opt)')
@click.option('--name', required=True, help='Base filename prefix')
@click.option('--model', type=click.Choice(['XGBC','RFC']), required=True)
@click.option('--upsampling', type=click.Choice(['none','smote','random']), default='none')
@click.option('--n-splits', default=5, help='CV folds for evaluation')
def train(output_dir, chisq, muvr, no_split,
          meta_file, features1, features2,
          name, model, upsampling, n_splits):
    """Run the full Train pipeline."""
    base = output_dir or datetime.now().strftime('%y%m%d_%H%M')
    os.makedirs(base, exist_ok=True)
    click.echo(f"Using base output dir: {base}")

    # Step 00: Split
    d0 = make_step_dir(base, 0, 'data_split')
    if not no_split:
        script = os.path.join(str(STEPS_DIR), '00_split_dataset.py')
        run_cmd(['python', script, '--meta-file', os.path.abspath(meta_file), '--out-prefix', name], cwd=d0)
    train_meta = os.path.join(d0, f"{name}_train.tsv")
    test_meta = os.path.join(d0, f"{name}_test.tsv")

    # Step 01: Chi²
    d1 = make_step_dir(base, 1, 'chisq_filtered')
    if chisq:
        script = os.path.join(str(STEPS_DIR), '01_chisq_selection.py')
        cmd = ['python', script, '--meta', os.path.abspath(train_meta),
               '--features1', os.path.abspath(features1), '--output_dir', d1]
        if features2:
            cmd += ['--features2', os.path.abspath(features2)]
        run_cmd(cmd)
        chisq_file = os.path.join(d1, f"{name}_top100000_features.tsv")
    else:
        chisq_file = None

    # Step 02: MUVR
    d2 = make_step_dir(base, 2, 'muvr_filtered')
    if muvr:
        if not chisq:
            click.echo("Error: --muvr requires --chisq", err=True)
            sys.exit(1)
        script = os.path.join(str(STEPS_DIR), '02_muvr_feature_selection.py')
        run_cmd(['python', script,
                 '--train_data', os.path.abspath(train_meta),
                 '--chisq_file', os.path.abspath(chisq_file),
                 '--model', model, '--output', d2])
        pat = os.path.join(d2, f"{name}_muvr_{model}_min.tsv")
        files = glob.glob(pat)
        if not files:
            click.echo(f"Error: no muvr_min found in {d2}", err=True)
            sys.exit(1)
        muvr_file = files[0]
    else:
        muvr_file = None

    # Step 03: Extract
    d3 = make_step_dir(base, 3, 'final_features')
    if muvr:
        script = os.path.join(str(STEPS_DIR), '03_extract_features.py')
        run_cmd(['python', script,
                 '--muvr_file', os.path.abspath(muvr_file),
                 '--chisq_file', os.path.abspath(chisq_file),
                 '--train_metadata', os.path.abspath(train_meta),
                 '--test_metadata', os.path.abspath(test_meta),
                 '--output_dir', d3])
        feat_train = os.path.join(d3, f"{name}_train.tsv")
        feat_test = os.path.join(d3, f"{name}_test.tsv")
    else:
        click.echo("Error: pipeline requires --muvr", err=True)
        sys.exit(1)

    # Step 04 & 05: Train + CV
    d4 = make_step_dir(base, 4, 'model')
    d5 = make_step_dir(base, 5, 'cv')
    script = os.path.join(str(STEPS_DIR), '04_train_model.py')
    run_cmd(['python', script,
             '--features', os.path.abspath(feat_train),
             '--label', 'outcome', '--group_column', 'group',
             '--model', model, '--sampling', upsampling,
             '--output_model', d4, '--output_cv', d5])
    mdl = os.path.join(d4, f"{name}_{model}_{upsampling}.joblib")

    # Step 06: Splits evaluation
    d6 = make_step_dir(base, 6, 'splits_evaluation')
    script = os.path.join(str(STEPS_DIR), '05_evaluate_model.py')
    run_cmd(['python', script,
             '--model', os.path.abspath(mdl),
             '--features', os.path.abspath(feat_test),
             '--label', 'outcome', '--group_column', 'group',
             '--n_splits', str(n_splits), '--output_dir', d6])

    click.echo("\n✅ Training pipeline done.")

@cli.command()
@click.option('--output-dir', default=None, help='Base output dir')
@click.option('--model-file', required=True, help='Trained model .joblib')
@click.option('--features', required=True, help='Features TSV')
@click.option('--name', required=True, help='Base name prefix')
@click.option('--label', default='outcome')
@click.option('--group-column', default='group')
def test(output_dir, model_file, features, name, label, group_column):
    """Run hold-out evaluation (no CV)."""
    base = output_dir or datetime.now().strftime('%y%m%d_%H%M')
    os.makedirs(base, exist_ok=True)
    click.echo(f"Using base output dir: {base}")
    d7 = make_step_dir(base, 7, 'test_evaluation')
    script = os.path.join(str(STEPS_DIR), '05_evaluate_model.py')
    run_cmd(['python', script,
             '--model', os.path.abspath(model_file),
             '--features', os.path.abspath(features),
             '--label', label, '--group_column', group_column,
             '--no_cv', '--output_dir', d7])
    click.echo("\n✅ Test pipeline done.")

if __name__ == '__main__':
    cli()

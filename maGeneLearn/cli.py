#!/usr/bin/env python3
"""
MaGeneLearn CLI – command reference
==================================

A convenience front‑end that orchestrates the numbered **MaGeneLearn** pipeline
scripts (`00` – `05`).  Two sub‑commands are exposed:

* **train** – build a model end‑to‑end (data split → optional feature‑selection → model fit → evaluation)
* **test**  – evaluate an existing model on an external feature matrix (no CV)

-------------------------------------------------------------------------------
TRAIN MODE
-------------------------------------------------------------------------------

```
magene-learn train [OPTIONS]
```

### Required inputs
| Option | File | Description |
|--------|------|-------------|
| `--meta-file` | `*.tsv` | Sample metadata with at least **`outcome`** (label) and **`group`** columns |
| `--features` | `*.tsv` | Primary k‑mer count matrix (samples × k‑mers) |
| `--name`      | *string* | Prefix used to label every output artefact |
| `--model`     | RFC \| XGBC | Choice of ML algorithm |

### Optional inputs / switches
| Option | Purpose |
|--------|---------|
| `--features2`            | Second k‑mer matrix merged with `--features` |
| `--features-train`       | Pre‑computed **training** feature table – bypasses Steps 00–03 |
| `--features-test`        | Pre‑computed **test** feature table; if omitted, Step 07 is skipped |
| `--chisq / --no-chisq`   | Run Chi² selection (Step 01) |
| `--muvr / --no-muvr`     | Run MUVR selection (Step 02) – *requires* `--chisq` |
| `--no-split`             | Skip Step 00; assumes existing `*_train.tsv` / `*_test.tsv` |
| `--upsampling`           | `none` (default) \| `smote` \| `random` |
| `--n-splits`             | Folds for cross‑validation (default **5**) |
| `--output-dir`           | Base directory (default: timestamp, e.g. `250704_1532`) |
| `--dry-run`              | Print planned shell commands but **do not execute** |

### Outputs (directory tree)
```
<output>/
├── 00_data_split/        *_train.tsv  *_test.tsv
├── 01_chisq/             <name>_top100000_features.tsv
├── 02_muvr/              <name>_muvr_<model>_min.tsv
├── 03_final_features/    <name>_train.tsv  <name>_test.tsv
├── 04_model/             <name>_<model>_<upsampling>.joblib
├── 05_cv/                cross‑validation metrics
├── 06_train_eval/        evaluation on training folds
└── 07_test_eval/         evaluation on hold‑out set (only if `--features-test`)
```

### Usage examples
```bash
# Full workflow: split → Chi² → MUVR → train (RFC, 5‑fold CV)
magene-learn train \
    --meta-file meta.tsv \
    --features kmers.tsv \
    --name exp \
    --model RFC \
    --chisq --muvr --upsampling smote

# Retrain using existing *_train/_test tables (no feature‑selection)
magene-learn train \
    --no-split \
    --meta-file meta.tsv \
    --features dummy.tsv \
    --name exp --model XGBC

# Feed final matrices directly; skip hold‑out evaluation
magene-learn train \
    --features-train final_train.tsv \
    --meta-file meta.tsv \
    --name exp --model RFC --n-splits 10
```

-------------------------------------------------------------------------------
TEST MODE
-------------------------------------------------------------------------------

```
magene-learn test --model-file <joblib> --features <tsv> --name <prefix> [OPTIONS]
```

### Required
| Option | Description |
|--------|-------------|
| `--model-file` | `.joblib` produced by **train** |
| `--features`   | Feature matrix to evaluate |
| `--name`       | Prefix for output artefacts |

### Optional
| Option | Default | Description |
|--------|---------|-------------|
| `--label`         | `outcome` | Column holding class labels |
| `--group-column`  | `group`   | Column used for grouped metrics |
| `--output-dir`    | timestamp | Base directory for outputs |

### Outputs
```
<output>/07_test_eval/  metrics tables & plots (no CV)
```

### Example
```bash
magene-learn test \
    --model-file 04_model/exp_RFC_none.joblib \
    --features external_validation.tsv \
    --name val
```

-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------
Add this `console_scripts` entry to **pyproject.toml** so that `pip install -e .`
or a normal build places a `magene-learn` executable on your `$PATH`:

```toml
[project.scripts]
magene-learn = "magenelearn_cli:cli"
```
"""
from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import List, Sequence

import click

# ---------------------------------------------------------------------------
# Locate numbered scripts
# ---------------------------------------------------------------------------
try:
    STEPS_DIR: Path = Path(resources.files("maGeneLearn.steps"))  # type: ignore[arg-type]
except (AttributeError, ModuleNotFoundError):
    STEPS_DIR = Path(__file__).resolve().parent / "steps"

# ---------------------------------------------------------------------------
# Context dataclass shared across steps
# ---------------------------------------------------------------------------
@dataclass
class Context:
    base_dir: Path
    name: str
    model: str  # "RFC" | "XGBC
    muvr_model: str
    upsample: str  # "none" | "smote" | "random"
    n_splits: int
    dry_run: bool = False
    label: str = "outcome"
    group_col: str = "group"
    n_iter: int = 100
    scoring: str = "balanced_accuracy"
    k: int = 100000
    lineage_col: str = "LINEAGE"


    # artefacts populated as we go
    train_meta: Path | None = None
    test_meta: Path | None = None
    chisq_file: Path | None = None
    muvr_file: Path | None = None
    feat_train: Path | None = None
    feat_test: Path | None = None
    model_file: Path | None = None
    full_matrix: Path | None = None


    # cache of numbered sub‑directories
    _step_dirs: dict[int, Path] = field(default_factory=dict, init=False, repr=False)

    def step_dir(self, idx: int, label: str) -> Path:
        if idx not in self._step_dirs:
            p = (self.base_dir / f"{idx:02d}_{label}").resolve()
            p.mkdir(parents=True, exist_ok=True)
            self._step_dirs[idx] = p
        return self._step_dirs[idx]

# ---------------------------------------------------------------------------
# Helper to run external scripts
# ---------------------------------------------------------------------------

def run(cmd: Sequence[str], *, cwd: Path | None, log: Path, dry: bool) -> None:
    click.echo(f"\n>>> {' '.join(cmd)} (cwd={cwd})")
    if dry:
        return

    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log.write_text(res.stdout + res.stderr)
    if res.returncode != 0:
        click.echo(res.stderr, err=True)
        click.echo(f"Step failed – see log {log}", err=True)
        sys.exit(res.returncode)

# ---------------------------------------------------------------------------
# Individual pipeline steps (still shelling out to 00–05 scripts)
# ---------------------------------------------------------------------------

def split(ctx: Context, meta_file: Path) -> None:
    d = ctx.step_dir(0, "data_split")
    script = STEPS_DIR / "00_split_dataset.py"
    run([
        sys.executable, str(script),
        "--meta-file", str(meta_file.resolve()),
        "--name", ctx.name,
        "--out-dir", str(d),
        "--lineage-col", ctx.lineage_col,
        "--group-col", ctx.group_col,  # optional but keeps column names consistent
        "--outcome-col", ctx.label,  # optional
        "--n-splits", str(ctx.n_splits)  # optional – matches CLI default
    ], cwd=d, log=d / "split.log", dry=ctx.dry_run)
    ctx.train_meta = d / f"{ctx.name}_train.tsv"
    ctx.test_meta = d / f"{ctx.name}_test.tsv"


def chisq(ctx: Context, features: Path, features2: Path | None) -> None:
    d = ctx.step_dir(1, "chisq")
    script = STEPS_DIR / "01_chisq_selection.py"
    cmd: List[str] = [sys.executable, str(script),
                      "--meta", str(ctx.train_meta.resolve()),
                      "--features1", str(features.resolve()),
                      "--output_dir", str(d.resolve()),
                      "--k", str(ctx.k),
                      "--name", ctx.name,
                      "--label", ctx.label,  # pass custom column names
                      ]
    if features2:
        cmd += ["--features2", str(features2.resolve())]
    run(cmd, cwd=d, log=d / "chisq.log", dry=ctx.dry_run)
    ctx.chisq_file = (d / f"{ctx.name}_top{ctx.k}_features.tsv").resolve()


def muvr(ctx: Context) -> None:
    if not ctx.chisq_file:
        click.echo("Error: --muvr requires Chi² step", err=True)
        sys.exit(1)

    d = ctx.step_dir(2, "muvr")
    script = STEPS_DIR / "02_muvr_feature_selection.py"

    # tmp folder that holds the Chi²-filtered training matrix for MUVR
    tmp_dir = (ctx.base_dir / "tmp_muvr_data").resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run([
        sys.executable, str(script),
        "--train_data", str(ctx.train_meta.resolve()),
        "--chisq_file", str(ctx.chisq_file.resolve()),
        "--model", ctx.muvr_model,
        "--output", str(d),
        "--group_col", ctx.group_col,
        "--outcome_col", ctx.label,
        "--filtered_train_dir", str(tmp_dir),
        "--name", ctx.name
    ], cwd=d, log=d / "muvr.log", dry=ctx.dry_run)

    matches = sorted(d.glob(f"{ctx.name}_muvr_{ctx.muvr_model}_min.tsv"))
    if not matches:
        click.echo("MUVR output not found", err=True)
        sys.exit(1)
    ctx.muvr_file = matches[0].resolve()


def extract_features(ctx: Context) -> None:
    """
    Build the final feature matrices.

    On the Chi² + MUVR branch it calls 03_extract_features.py, forwarding
    the correct label/group names and *optionally* the test-metadata file.
    """
    # -------------------------------------------------------- short-circuit
    if ctx.feat_train:          # user already supplied final matrices
        return

    d = ctx.step_dir(3, "final_features")
    script = STEPS_DIR / "03_extract_features.py"

    # -------------------------------------------------- Chi² + MUVR branch
    if ctx.muvr_file:
        cmd = [
            sys.executable, str(script),
            "--muvr_file",      str(ctx.muvr_file),
            "--chisq_file",     str(ctx.full_matrix),
            "--train_metadata", str(ctx.train_meta),
            "--output_dir",     str(d),
            "--group_column",   ctx.group_col,
            "--label",          ctx.label,
            "--name",           ctx.name,
        ]

        # only add --test_metadata when we actually have one
        if ctx.test_meta:
            cmd += ["--test_metadata", str(ctx.test_meta)]

        run(cmd, cwd=d, log=d / "extract.log", dry=ctx.dry_run)

        # store absolute paths for downstream steps
        ctx.feat_train = (d / f"{ctx.name}_train.tsv").resolve()
        if ctx.test_meta:
            ctx.feat_test = (d / f"{ctx.name}_test.tsv").resolve()

    # --------------------------------------------- Chi² without MUVR → error
    elif ctx.chisq_file:
        click.echo("Chi² without MUVR not supported in this pipeline.", err=True)
        sys.exit(1)

    # --------------------------------------- no feature-selection branch
    else:
        ctx.feat_train = ctx.train_meta.with_name(
            ctx.train_meta.name.replace("_train.tsv", "_full_features.tsv")
        ).resolve()

        if ctx.test_meta and ctx.test_meta.exists():
            ctx.feat_test = ctx.test_meta.with_name(
                ctx.test_meta.name.replace("_test.tsv", "_full_features.tsv")
            ).resolve()

def train_model(ctx: Context) -> None:
    d_model = ctx.step_dir(4, "model")
    d_cv = ctx.step_dir(5, "cv")
    script = STEPS_DIR / "04_train_model.py"
    run([
        sys.executable, str(script),
        "--features", str(ctx.feat_train),
        "--model", ctx.model, "--sampling", ctx.upsample,
        "--output_model", str(d_model),
        "--output_cv", str(d_cv),
        "--name", ctx.name,
        "--label", ctx.label,
        "--group_column", ctx.group_col,
        "--n_iter", str(ctx.n_iter),
        "--scoring", ctx.scoring
    ], cwd=d_model, log=d_model / "train.log", dry=ctx.dry_run)
    ctx.model_file = d_model / f"{ctx.name}_{ctx.model}_{ctx.upsample}.joblib"


def evaluate_train(ctx: Context) -> None:
    d = ctx.step_dir(6, "train_eval")
    script = STEPS_DIR / "05_evaluate_model.py"
    run([
        sys.executable, str(script),
        "--model", str(ctx.model_file),
        "--features", str(ctx.feat_train),
        "--n_splits", str(ctx.n_splits),
        "--output_dir", str(d),
        "--name", ctx.name + "_train",
        "--label", ctx.label,
        "--group_column", ctx.group_col
    ], cwd=d, log=d / "eval_train.log", dry=ctx.dry_run)


def evaluate_holdout(ctx: Context) -> None:
    if not ctx.feat_test:
        return  # nothing to do
    d = ctx.step_dir(7, "test_eval")
    script = STEPS_DIR / "05_evaluate_model.py"
    run([
        sys.executable, str(script),
        "--model", str(ctx.model_file),
        "--features", str(ctx.feat_test),
        "--no_cv",
        "--output_dir", str(d),
        "--name", ctx.name + "_test",
        "--label", ctx.label,
        "--group_column", ctx.group_col
    ], cwd=d, log=d / "eval_test.log", dry=ctx.dry_run)

# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

@click.group()
@click.option("--dry-run", is_flag=True, help="Print commands without executing.")
@click.pass_context
def cli(ctx: click.Context, dry_run: bool) -> None:
    ctx.obj = {"dry": dry_run}

# -------------- Train command ----------------

@cli.command()
@click.option("--meta-file", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--train-meta", type=click.Path(exists=True, path_type=Path),
              help="Pre-split training metadata TSV (implies --no-split)")
@click.option("--test-meta",  type=click.Path(exists=True, path_type=Path),
              help="Pre-split test metadata TSV (optional)")
@click.option("--lineage-col", default="LINEAGE", help="Column holding lineage/clade assignments (used only by step 00 when the split is executed).")
@click.option("--features", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--features2", type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--model", type=click.Choice(["XGBC", "RFC"]), required=True, help="Classifier used in the final training step (04_train_model.py)")
@click.option("--muvr-model","muvr_model", type=click.Choice(["XGBC", "RFC"]), default=None, help="Classifier used *inside* the MUVR feature-selection step ""(defaults to the value of --model)")
@click.option("--upsampling", type=click.Choice(["none", "smote", "random"]), default="none")
@click.option("--n-splits", "n_splits", default=5, show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path))
@click.option("--chisq/--no-chisq", "chisq_flag", default=False)
@click.option("--muvr/--no-muvr", "muvr_flag", default=False)
@click.option("--no-split", is_flag=True, help="Skip dataset split step")
@click.option("--features-train", type=click.Path(exists=True, path_type=Path))
@click.option("--features-test", type=click.Path(exists=True, path_type=Path))
@click.option("--label", default="outcome", show_default=True)
@click.option("--group-column", "group_column", default="group", show_default=True)
@click.option("--n-iter", "n_iter", type=int, default=100,
              help="Number of iterations for RandomizedSearchCV")
@click.option("--k", type=int, default=100000,
               help="Number of top features to keep in the Chi² step")
@click.option("--scoring", default="balanced_accuracy", show_default=True, type=click.Choice(["accuracy", "balanced_accuracy", "f1", "f1_macro", "f1_micro", "precision", "recall", "roc_auc"]),
              help="Metric used to pick the best hyper-parameters in 04_train_model.py")
@click.option("--chisq-file", type=click.Path(exists=True, path_type=Path))
@click.pass_context

def train(click_ctx: click.Context, *,
          # metadata
          meta_file:    Path | None,
          train_meta:   Path | None,
          test_meta:    Path | None,
          no_split:     bool,
          #split
          lineage_col: str,
          # features
          features:    Path | None,
          features2:    Path | None,
          features_train: Path | None,
          features_test:  Path | None,
          # modelling + FS flags
          chisq_flag:   bool,
          chisq_file:   Path | None,
          muvr_flag:    bool,
          k:            int,
          # core settings
          name:         str,
          model:        str,
          muvr_model:   str,   # fallback
          upsampling:   str,
          n_splits:     int,
          n_iter:       int,
          scoring:      str,
          label:        str,
          group_column: str,
          output_dir:   Path | None) -> None:
    """Train model end-to-end, with optional Chi² + MUVR feature selection."""

    # ------------------------------------------------------------------ sanity
    if bool(meta_file) == bool(train_meta):
        raise click.UsageError(
            "Supply EITHER --meta-file (unsplit) OR --train-meta (pre-split)."
        )

    if chisq_file and chisq_flag:
        raise click.UsageError("--chisq and --chisq-file are mutually exclusive.")

    if muvr_flag and not (chisq_flag or chisq_file):
        raise click.UsageError("--muvr requires --chisq or --chisq-file.")

    if muvr_flag and features is None:
        raise click.UsageError(
            "--muvr also needs the *full* feature matrix via --features "
            "so that Step 03 can extract the selected k-mers."
        )



    # ----------------------------------------------------------------- context
    base = (output_dir or
            Path(datetime.now().strftime("%y%m%d_%H%M"))).resolve()
    base.mkdir(parents=True, exist_ok=True)
    click.echo(f"Base output dir: {base}")

    ctx = Context(
        base_dir   = base,
        name       = name,
        model      = model,
        upsample   = upsampling,
        n_splits   = n_splits,
        dry_run    = click_ctx.obj["dry"],
        label      = label,
        group_col  = group_column,
        n_iter     = n_iter,
        scoring    = scoring,
        k          = k,
        muvr_model=muvr_model or model, # fallback
        lineage_col = lineage_col,
    )

    # -------------------------------------------- ingest pre-existing artefacts
    if chisq_file:
        ctx.chisq_file = chisq_file.resolve()

    if features_train:
        ctx.feat_train = features_train.resolve()

    if features_test:
        ctx.feat_test = features_test.resolve()

    if train_meta:
        ctx.train_meta = train_meta.resolve()
        no_split = True          # implicit
    if test_meta:
        ctx.test_meta  = test_meta.resolve()
        no_split = True

    if features:
        ctx.full_matrix = features.resolve()

    if muvr_flag and ctx.full_matrix is None:
        raise click.UsageError(
            "--muvr needs the ORIGINAL k-mer matrix via --features "
            "(even if you hand in --features-train)."
        )

    # ------------------------------------------------------------------ plan
    plan: List[tuple[bool, callable[[Context], None]]] = []

    # ---------------- step 00 – split (only if we still need train_meta)
    if ctx.train_meta is None and not features_train:
        if no_split:
            # derive legacy filenames (<name>_train.tsv) for back-compat
            ctx.train_meta = meta_file.with_name(f"{name}_train.tsv").resolve()
            ctx.test_meta  = meta_file.with_name(f"{name}_test.tsv").resolve()
        else:
            plan.append((True, lambda c: split(c, meta_file)))

    # ---------------- step 01 – Chi²
    if chisq_flag:
        plan.append((True, lambda c: chisq(c, features, features2)))

    # ---------------- step 02 – MUVR
    if muvr_flag:
        plan.append((True, muvr))

    # ---------------- downstream steps
    plan.extend([
        (True, extract_features),
        (True, train_model),
        (True, evaluate_train),
        (ctx.feat_test is not None, evaluate_holdout),
    ])

    # ---------------------------------------------------------------- execute
    for cond, func in plan:
        if cond:
            func(ctx)

    click.echo("\n✅ Training pipeline complete.")




# -------------- Test command -----------------

@cli.command()
@click.option("--model-file", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--features-test", "ready_features",type=click.Path(exists=True, path_type=Path),required=False,help="Pre-filtered test feature table; evaluated as-is.",)
@click.option("--name", required=True)
@click.option("--label", default="outcome")
@click.option("--group-column", default="group")
@click.option("--output-dir", type=click.Path(path_type=Path))
#extract features
@click.option("--features","full_features",type=click.Path(exists=True, path_type=Path),required=False,help="Full k-mer count matrix (will be filtered by --muvr-file).")
@click.option("--test-metadata",  type=click.Path(exists=True, path_type=Path), help="Metadata TSV for the external test set (required with --extract-features)")
@click.option("--muvr-file",  type=click.Path(exists=True, path_type=Path), help="*_muvr_*_min.tsv file with selected features(required with --extract-features)")
@click.option("--predict-only", is_flag=True, help="Only output predictions without computing performance metrics.")
@click.pass_context
def test(click_ctx: click.Context, *,
         model_file: Path,
         full_features:  Path | None,
         ready_features: Path | None,
         name: str,
         label: str,
         group_column: str,
         output_dir: Path | None,
         muvr_file: Path | None,
         test_metadata : Path | None,
         predict_only) -> None:

    #1. Sanity check
    if (full_features is None) == (ready_features is None):
        # Either both missing OR both given – both invalid
        raise click.UsageError(
            "Supply exactly one of --features (full matrix) "
            "or --features-test (ready feature table)."
        )

    if full_features and not muvr_file:
        raise click.UsageError("--features also needs --muvr-file")

    if full_features and not predict_only and not test_metadata:
        raise click.UsageError("--features also needs --test-metadata unless using --predict-only")

    # if full_features and not (muvr_file and test_metadata):
    #     raise click.UsageError(
    #         "--features also needs --muvr-file and --test-metadata."
    #     )

    #2 - Basic context
    base = output_dir or Path(datetime.now().strftime("%y%m%d_%H%M"))
    base.mkdir(parents=True, exist_ok=True)

    ctx = Context(base,
                  name,
                  model="NA",
                  muvr_model="None",
                  upsample="none",
                  n_splits=0,
                  dry_run=click_ctx.obj["dry"],
                  label=label,
                  group_col=group_column)
    ctx.model_file = model_file.resolve()

    # ── 3. branch A – build table from full matrix ─────────────────────────
    #ctx.full_matrix = features.resolve()

    if full_features:
        ctx.full_matrix = full_features.resolve()

        d = ctx.step_dir(3, "final_features")
        script = STEPS_DIR / "03_extract_features.py"

        cmd = [
            sys.executable, str(script),
            "--muvr_file", str(muvr_file.resolve()),
            "--chisq_file", str(ctx.full_matrix),
            "--output_dir", str(d),
            "--name", name,
        ]

        if not predict_only:
            if test_metadata is None:
                raise click.UsageError("--test_metadata required unless using --predict-only")
            cmd.extend([
                "--test_metadata", str(test_metadata.resolve()),
                "--label", label,
                "--group_column", group_column,
            ])

        run(cmd, cwd=d, log=d / "extract_test.log", dry=ctx.dry_run)
        ctx.feat_test = (d / f"{name}_test.tsv").resolve()
        # run([
        #     sys.executable, str(script),
        #     "--muvr_file", str(muvr_file.resolve()),
        #     "--chisq_file", str(ctx.full_matrix),
        #     "--test_metadata", str(test_metadata.resolve()),
        #     "--output_dir", str(d),
        #     "--label", label,
        #     "--group_column", group_column,
        #     "--name", name
        # ], cwd=d, log=d / "extract_test.log",
        #     dry=ctx.dry_run)

        # ctx.feat_test = (d / f"{name}_test.tsv").resolve()

        # ── 4. branch B – ready table supplied ---------------------------------
    else:
        ctx.feat_test = ready_features.resolve()

        # ── 5. evaluate --------------------------------------------------------

    if predict_only:
        d = ctx.step_dir(7, "test_eval")
        script = STEPS_DIR / "05_evaluate_model.py"
        run([
            sys.executable, str(script),
            "--model", str(ctx.model_file),
            "--features", str(ctx.feat_test),
            "--no_cv",
            "--predict_only",
            "--output_dir", str(d),
            "--name", ctx.name + "_test",
        ], cwd=d, log=d / "predict.log", dry=ctx.dry_run)
    else:
        evaluate_holdout(ctx)

    #evaluate_holdout(ctx)
    click.echo("\n✅ Test evaluation complete.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli()

# Repeatable validation

From the repository root, activate an environment containing the pinned
`setup.py` dependencies and install the checkout with `python -m pip install -e .`.

```bash
# Complete declared workflow coverage; this is the default.
python scripts/validate.py

# Faster feedback during development.
python scripts/validate.py --profile quick

# Optional explicit output path; it must not already exist.
python scripts/validate.py --profile quick --output validation_results/my_change

# Focused fixture, hold-out scheduling, and validator regression tests.
python -m unittest discover -s tests -p 'test_*.py' -v
```

The runner uses only committed fixtures under `test/`. Source uploads,
pre-existing models, temporary directories from earlier development, and
machine-specific Python paths are not required. Original source paths in the
fixture manifest remain provenance only. No fixtures are rebuilt or rewritten.

## Coverage

Both profiles now run the matrix twice: the original TSV route and a fresh CSR
route. Quick requires 40 successful models per representation (80 total), and
full requires 240 configurations per representation (480 total), plus six saved
model variants per representation. Coverage and independent artifact audits are
reported separately as `matrix`/`artifact_audit` and
`csr_matrix`/`csr_artifact_audit`. The original narrowly scoped ENN exception
remains unchanged; it is not automatically extended to additional sparse cases.
CSR runs include eight fresh selectors, four full training workflows, all four
labelled/unlabelled ready/extraction routes, sampler endpoints and model branches,
and full-profile SHAP/SVM importance diagnostics. Header-only TSV feature lists
remain supported alongside sparse matrix inputs.

`test_matrix_io.py` checks portable bundles, checksum/identifier corruption,
count preservation, conversion modes, and dense allocation limits.
`test_sparse_input.py` checks all 24 classifier/sampler combinations on CSR for
both binary and multiclass outcomes,
including assertions that samplers receive CSR before any XGBoost dense adapter,
dense-zero XGBoost probability parity, save/load and shuffled feature alignment,
and independent sparse Chi-square presence-statistic equivalence on count-valued
inputs. Conversion does not modify the committed fixtures.

To diagnose the sparse matrix alone (not a substitute for the canonical profile):

```bash
python scripts/validation/matrix.py --input-format csr \
  --output validation_results/csr_diagnostic --skip-explanations
```

| Check | Quick | Full |
|---|---|---|
| Fixture integrity, focused regressions, three CLI help commands | Required | Required |
| Binary/multiclass model × selection matrix | 40 cases, sampling none | 240 cases, all six samplers |
| Fresh validated selectors | Eight, computed/supplied Chi-square | Eight, computed/supplied Chi-square |
| Full CLI train: both outcomes × automatic/supplied splits | Four | Four |
| Reserved labelled/unlabelled ready/extraction routes | Every matrix model | Every successfully trained matrix/variant model |
| Fixed-trial model branches, sampler endpoints, invalid inputs | Required | Required |
| Missing-feature zero-fill, unseen labels, empty Boruta | Required | Required |
| Additional CLI LR l1/elasticnet, XGBC lossguide | Omitted | Six runs |
| LR coefficients and feature importance | Required | Required |
| Tree SHAP | Explicitly omitted | Required, including NPY/TSV alignment |
| SVM permutation importance | Explicitly omitted | Two full-feature diagnostic runs |
| Independent saved-artifact audits | 40 | 246, or 245 after the one verified rejection below |

Models are RFC/XGBC/SVM/LR. Strategies are no selection or Chi-square followed
by MUVR/Boruta using RFC/XGBC. Sampling settings are none, random, smote,
random_under, enn, and smoteenn. Supplied selector matrices come from training
data in the current run; validated outputs are reused only within that run.

Budgets are CV=3, optimization trials=3, jobs=1, Chi-square k=256, MUVR
repetitions/outer/inner=1/3/3, and Boruta maximum 100 iterations. Three searched
trials do not guarantee branch coverage: fixed-trial tests force both SVM
kernels, all LR penalties, and both XGBoost policies separately. Nondefault CLI
variants cover no-selection/sampling-none, not their entire Cartesian product.

SVM interpretation retains all 2,471 predictors and the evaluator's existing
three CV folds/ten repeats. It uses deterministic training-only subsets of
36 multiclass and 24 binary samples, with six samples per source/lineage.
Full-training-sample permutation importance is not part of either profile.

## What constitutes a pass

The runner checks artifacts and scientific invariants, not only exit codes.
It checks IDs, groups, classes, feature order/values, encoder preservation,
probability alignment/range/sums, independently recomputed metrics,
save/load parity, four prediction routes, and applicable explanations. Artifact
audits also check saved classifier/sampler configuration and optimization output,
including independently recomputed fold-score means and standard deviations.

Focused hold-out tests check generated/supplied tables, no-hold-out execution,
and selection-only execution. An isolated in-memory copy of the original
scheduling behavior must fail the generated-hold-out assertion. Four real
training workflows must also produce correct hold-out evaluation artifacts.
Deliberate missing-artifact and altered-probability tests verify that the
validation assertions themselves can detect failures.

No accuracy threshold or exact selected mask is prescribed. These datasets
exercise functionality, not independent predictive performance. Selection
precedes downstream CV and tuned evaluation is not nested CV. MUVR receives
neither an explicit seed nor a group vector. Seeds remain 42 for split/Boruta
and 50 for training/evaluation. Count-valued inputs, every scoring/error branch,
all Python/OS versions, and legacy artifact variants are not exhaustively tested.

## Results and failure handling

Each run creates a unique ignored `validation_results/` directory containing:

- `summary.md`, `report.json`, and `junit.xml`.
- Commands, logs, exit codes, timings, dependency versions, source/fixture hashes,
  working-tree state, seeds, and concurrency.
- Per-stage reports and generated models.
- `models_for_dataset3.tsv`, pairing successful models with compatible ready
  tables, raw inputs, feature specifications, and checksums.
- `failed_cases/` with matrix-model failure evidence for GitHub uploads.
- `expected_rejections/` with the independently verified ENN rejection, when applicable.

Passed, expected-rejection, failed, blocked, and deliberately omitted checks remain distinct.
Missing/duplicate matrix cases, missing artifacts, failed selectors, timeouts,
or incomplete required stages make the command return nonzero. Quick success
means only that the quick profile passed. Failed reports are never converted
to passes by silently lowering budgets or rewriting expectations.

### One permitted ENN rejection

Full validation permits `expected_rejection` only for multiclass / `muvr_RFC` /
`XGBC` / `enn`. The training command must exit one with a terminal exhausted-ENN
traceback and produce no model or evaluation/prediction outputs. The validator
independently reconstructs the label encoding and three grouped folds (seed 50),
then directly applies ENN with `kind_sel="mode"` for each neighbour count from
the requested value down to one. Every candidate must lose a class in at least
one training fold or the final-fit partition. A candidate that preserves all
classes invalidates the claimed rejection. No hold-out outcomes are used.

Evidence includes commands, exit code, selected features, ordered sample IDs,
group/fold membership, input/log hashes and all before/after counts. It is kept
in `expected_rejection.json` and copied under `expected_rejections/`; the existing
GitHub artifact rules upload these JSON reports and logs.

The matrix still requires 240 distinct configurations. A valid result can be
239 trained configurations, one verified expected rejection and zero unexpected
failures. Reports never call that 240 trained models. The rejected case has no
model-index entry, downstream predictions or explanations, so required artifact
audits become 245 (239 main models plus six variants). The allowance comes from
the verified rejection record, never from the number of artifacts found.

If that configuration trains successfully, all normal checks and all 246 audits
apply. MUVR may select different features across runs. Other failures and missing
outputs remain failures; the exception never applies to quick validation. JUnit
records the verified rejection as a successful behavioral check with an explicit
`outcome=expected_rejection` property, not a skip. Validation JSON schema version
2 adds separate trained/expected-rejection/accepted/failure counts; the legacy
coverage `passed` field continues to mean successfully trained configurations.

Each CLI subprocess is limited to 30 minutes, with a two-hour limit for each
SVM importance diagnostic. Timed-out subprocess groups are terminated so their
optimization workers do not continue running. Numerical-library/joblib workers
are bounded to one; profile stages run serially. Record actual runtime rather
than treating these limits as performance targets.

## GitHub Actions

The Validation workflow uses Ubuntu 22.04/Python 3.9, installs the checkout and
its pinned dependencies, and invokes this same entry point. It runs quick
validation on pull requests targeting main and pushes to main. There is no
schedule. Use Actions → Validation → Run workflow to choose full or quick;
manual runs default to full. GitHub shows the Run workflow button once the
workflow exists on the default branch. Before merge, a registered workflow can
also be dispatched through the GitHub API with the feature branch as its ref.

Only dependency downloads are cached. Reports/logs are uploaded even when
validation fails and retained for seven days. Generated models and explanation
tables are uploaded for failed matrix cases, rather than all successful models.
Successful model paths in the downloadable index refer to the runner's working
directory; regenerate locally to use those successful models after a hosted run.

The job summary links to run logs and downloadable evidence. Branch protection
and automatic merging are not configured by this workflow. Hosted execution
must be reported separately from local execution; adding YAML alone does not
prove the workflow passed.

## Extending coverage

For each bug fix, add a focused regression test demonstrating the failure.
For a new model, selector, sampler, or input route, extend the shared case lists,
explicit expected coverage, artifact assertions, and this coverage table.
Keep negative cases separate from valid supported configurations: a currently
broken accepted configuration is a failure, not an unsupported option.
The independently verified ENN rejection above is the single documented exception.

Before claiming full validation, run the default profile from a fresh checkout
without old models or local source uploads, inspect its complete report, and
record any unexecuted hosted checks. Preserve datasets and original failure
evidence throughout development.

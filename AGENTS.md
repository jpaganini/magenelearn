# maGeneLearn development instructions

maGeneLearn is scientific software for bacterial-genomics classification.
Prioritize scientific correctness, reproducibility, comprehensive validation,
data integrity, and backward compatibility.

## Repository map and conventions

- `maGeneLearn/cli.py` implements the Click `maGeneLearn train` and
  `maGeneLearn test` commands and subprocess orchestration.
- `maGeneLearn/steps/00_...` through `05_...` implement splitting, streaming
  Chi-square filtering, MUVR/Boruta selection, feature extraction, Optuna
  training, and evaluation/prediction.
- `setup.py` defines the executable, Python requirement, and dependency pins;
  `pyproject.toml` configures setuptools. Consult README.md for usage.
- Preserve existing CLI options, defaults, numbered output directories,
  filenames, TSV schemas, and supported model-artifact formats unless a
  justified compatibility change is explicitly in scope.
- Follow nearby function, argparse/Click, logging, and error-handling patterns.
  Make focused changes; avoid unrelated refactoring and new tooling.
- Preserve streaming and selected-column reads for wide genomic matrices.
  Do not introduce unnecessary full-matrix copies or nested parallelism.
- Update relevant documentation when behavior changes. Verify examples against
  source and actual fixture paths. Do not change dependency pins casually.

## Scientific correctness and data integrity

- Preserve original inputs. Write generated results separately, using unique
  output directories. Do not overwrite reference datasets or silently rewrite
  expected results to make tests pass.
- Validate sample and feature identifiers, duplicates, required columns, missing
  values, numeric ranges, and sample/feature/label/group alignment.
- Track sample counts and identifiers through every intersection, exclusion,
  deduplication, and transformation. Report changes in dataset composition.
- Preserve class labels and group assignments. Decode predictions with the
  training artifact's encoder; align probability columns with fitted classes.
  Handle unseen labels explicitly rather than silently assigning a known class.
- Preserve existing semantics unless a methodological change is justified:
  Chi-square uses `value > 0`; top-k features retain source order; selector
  preparation keeps the last sorted sample per `(group, outcome)`; extraction
  uses inner joins and may zero-fill missing selected features.
- Check integer casts for overflow and unintended value changes. Do not assume
  count-valued inputs behave identically to binary fixtures.
- Preserve metric definitions, averaging, class order, and supported-class
  handling. Validate binary and multiclass behavior independently. Distinguish
  average precision/AUPRC from ROC AUC and other integration definitions.
- Changes to splitting, selection, weighting, sampling, scoring, encoding,
  missing-data policy, or statistical reporting require explicit scientific
  justification and appropriate before/after validation. Seek maintainer input
  when the intended methodology cannot be established.

## Sparse CSR inputs and outputs

- Accept feature matrices as existing TSV files or portable CSR bundles through
  the existing feature-matrix options. A bare `.npz` is insufficient: bundles
  contain `matrix.npz`, ordered `samples.tsv` and `features.tsv`, and a versioned
  `manifest.json` with checksums, shape, dtype and provenance.
- Use `maGeneLearn/matrix_io.py` for shared loading, validation and writing;
  `scripts/prepare_sparse_input.py` assesses inputs and optionally converts them.
  Preserve original inputs, numeric values, integer/float dtype behavior, and
  sample/feature order. Do not silently binarize counts or overflow integer casts.
- Write generated bundles under the existing numbered output directories with
  `.csr` bundle names. Keep TSV output names for TSV workflows and evaluation
  tables. Bundles can be reused through the corresponding feature-matrix options;
  reuse supervised selection only with verified training-partition provenance.
- Native CSR resampling is the approved sparse behavior. Keep all samplers inside
  the fitted pipeline and operating on CSR. Equal-distance neighbor ties may
  change retained IDs, synthetic samples and predictions relative to dense input;
  do not require dense membership/prediction equality or densify sampling to force
  it. Independently verify class retention, alignment and metrics in both routes.
  Preserve ENN fallback checks and class-loss failures; native CSR does not permit
  changing samplers, dropping classes or broadening the rejection exception.
- RFC, SVM and LR use sparse fitting inputs. XGBoost converts to dense after
  sampling to preserve observed-zero semantics. MUVR/Boruta preparation and
  explanation routes also have bounded dense operations. Honor
  `MAGENELEARN_MAX_DENSE_MB` (default 1024 MiB), preserve feature alignment, and
  report these limits; this setting is not a total process memory cap.
- Record input checksums, feature/sample order, CV membership, seeds, concurrency,
  resampling policy and dependency versions in new model artifacts. Preserve
  compatibility with existing plain and dictionary artifacts.

## Leakage prevention and reproducibility

- Fit learned preprocessing, supervised feature selection, and resampling only
  on the training partition appropriate to each evaluation.
- Keep resampling inside the fitted imbalanced-learn pipeline.
- Never use final evaluation data to choose features, hyperparameters, thresholds,
  or preprocessing parameters. Verify provenance of supplied selection artifacts.
- Check sample and intended group separation across splits and CV folds.
  Initial splitting operates within lineages; establish whether groups are
  nested within lineages. Do not describe this as holding whole lineages out.
- Current feature selection precedes downstream CV, and evaluation reuses tuned
  hyperparameters. Do not claim this is nested CV or an unbiased evaluation of
  the complete selection-and-tuning procedure.
- Do not assume MUVR internal folds are group-aware: this repository passes no
  group vector to its fit call.
- Preserve and record existing seeds: initial split/Boruta use 42;
  training/evaluation use 50. Propagate seeds consistently in changed stochastic
  code without silently changing established defaults.
- Record inputs/checksums, sample/feature/class order, split membership,
  configuration, seeds, dependency versions, and concurrency for reproducible
  runs. Use serial optimization for deterministic reference comparisons.
- MUVR receives no explicit seed here, and parallel optimization may vary.
  Report these limits; do not promise determinism without checking it.

## Mandatory validation

During development, run relevant unit tests. For substantial changes affecting
data preparation, selection, training, tuning, evaluation, prediction, artifacts,
or orchestration, comprehensive integration validation is mandatory before
calling the change fully validated.

The logical matrix is every final model `{RFC, XGBC, SVM, LR}` crossed with:

1. No feature selection.
2. Chi-square -> MUVR using RFC.
3. Chi-square -> MUVR using XGBC.
4. Chi-square -> Boruta using RFC.
5. Chi-square -> Boruta using XGBC.

This is 20 logical combinations. Use an explicit RFC/XGBC `--feature-model` for
SVM/LR selection runs. Chi-square alone and simultaneous MUVR/Boruta are invalid.

- Cover computed Chi-square and supplied `--chisq-file` routes; no-selection
  training must exercise actual ready feature tables.
- Cover all six sampling settings: `none`, `random`, `smote`, `random_under`,
  `enn`, `smoteenn` (120 logical model/selection/sampling configurations).
- Include LR l1/l2/elasticnet and XGBoost depthwise/lossguide branches.
  Exercise both searched SVM kernels in targeted tests.
- Cover binary/multiclass labels, automatic/supplied splits, selection-only runs,
  ready feature tables, save/load, labelled hold-out/external evaluation, and
  unlabelled prediction through ready-table and extraction routes.
- Check MUVR min/mid/max exports and automatic min consumption; check Boruta
  confirmed-feature output and its empty-selection failure.
- Exercise tree SHAP, LR coefficients, full-feature SVM permutation importance,
  and supported skip/failure behavior. Preserve previously fixed class/probability,
  binary AUPRC, SHAP alignment, filename-parsing, and SVM alignment behavior.
- Test invalid combinations and inputs for clear failures. A currently broken
  accepted configuration is a failure, not an unsupported configuration. The
  single independently verified ENN rejection described below is the only
  explicit exception; it is not successful model training.

Use small, documented binary and multiclass fixtures with adequate independent
groups and per-fold class counts for all configured neighbor-based samplers.
Use fixed seeds, explicit concurrency, bounded optimization budgets, and reusable
fixtures. Reduce data size and search budgets without removing methods, changing
selection semantics, or weakening scientific assertions. Cache selector outputs
for downstream tests only after validating the corresponding selection run.

A zero exit status is insufficient. Verify expected artifacts, nonempty selected
features, sample coverage, group separation, feature order, encoder preservation,
probability shape/order/range/row sums, prediction-label validity, confusion-matrix
totals, and independently recomputed metrics. Verify save/load prediction parity.
Check explanation dimensions, class/sample/feature alignment, and reported status.

Validation commands and infrastructure:
- Run commands from the repository root with the pinned dependencies and the
  current checkout installed using `python -m pip install -e .`.
- Run focused regression tests during development:
  `python -m unittest discover -s tests -p 'test_*.py' -v`.
- Run quick integration validation before presenting pipeline changes for review:
  `python scripts/validate.py --profile quick`.
- Run complete declared workflow validation for substantial pipeline changes:
  `python scripts/validate.py`.
- Before claiming full validation, run the full profile in a fresh, isolated
  checkout containing the exact candidate changes, without local uploads,
  previous models or selector outputs from earlier runs. Preserve working-tree
  changes.
- Targeted reruns support diagnosis but do not replace required profile runs.

Coverage and supporting scripts:
- `scripts/validate.py` is the canonical local and GitHub Actions entry point.
- Validate TSV and CSR independently. Quick covers 40 model/selection cases per
  representation (80 total), across both outcome types with sampling `none`.
  Full covers 240 configurations per representation (480 total), plus six saved
  model variants per representation (12 total).
- Both profiles include fixture integrity, regression tests and CLI help. Each
  representation requires eight fresh selectors, four complete training
  workflows, and labelled/unlabelled testing through ready-table and extraction
  routes. Convert existing fixtures into new CSR bundles; do not alter `test/`.
- `tests/test_matrix_io.py` and `tests/test_sparse_input.py` cover bundle integrity,
  count/dtype preservation, alignment, dense boundaries and all model/sampler
  pairs for binary/multiclass CSR inputs. Synthetic unit inputs are not new
  independent biological validation datasets.
- For an explicitly requested CSR-only run, use the supporting runners with
  `--input-format csr`; for example:
  `python scripts/validation/matrix.py --input-format csr --samplings none random smote random_under enn smoteenn --skip-svm-importance --output validation_results/csr_full_matrix`.
  Run the applicable CSR runtime/variant stages and independent artifact audit
  separately. Document exact executed coverage and outstanding stages; a scoped
  run does not establish a new full canonical profile pass. Do not rerun TSV
  workflows when the maintainer explicitly limits the run to CSR.
- Full additionally requires tree SHAP and full-feature SVM permutation
  importance diagnostics on the documented 36/24-sample subsets.
- `tests/` contains automated regression tests; `scripts/validation/` contains
  supporting runners and artifact audits. See `tests/README.md` for details.
- Extend cases, expected coverage and scientific assertions when adding a
  supported model, selector, sampler or workflow route.

Results and failure handling:
- Preserve `test/` fixtures. Write generated results to a unique directory under
  ignored `validation_results/`; an explicit `--output` directory must be new.
- Inspect `summary.md`, `report.json`, `junit.xml` and relevant stage logs.
  Missing artifacts, incomplete coverage, failed required cases and timeouts
  prevent a validation pass.
- Preserve failure evidence. Do not weaken budgets, change seeds, rewrite
  expectations or silently substitute methods to obtain a passing result.
- For sampling changes, verify class retention in every training fold and final
  fit, and preservation of applied settings in trial tables and saved pipelines.
- Distinguish successful fallback from exhausted fallback. Only the full-profile
  multiclass / `muvr_RFC` / `XGBC` / `enn` case may be `expected_rejection`:
  training must exit one with an exhausted-ENN traceback, produce no model or
  evaluation artifacts, and independently reproduce class loss for every
  candidate from the requested neighbour count down to one across all three
  grouped training folds and the final-fit partition (seed 50). Do not call the
  production fallback helper for this verification. Record IDs, feature order,
  input/log checksums, commands, exit code and before/after class counts.
- Successful execution of that case must pass the normal model checks. Other
  errors, timeouts, unverifiable claims and other configurations remain failed.
  At most one verified rejection is allowed; all 240 distinct full-profile
  configurations remain required. Report trained models and expected rejections
  separately. JUnit treats a verified rejection as a successful behavioral check,
  with an explicit outcome property, never as skipped or successfully trained.
- Full requires 246 artifact audits per representation, with a reduction of one
  only in the representation containing the single independently verified
  rejection allowed above. Do not grant a second exception for CSR. Derive the
  allowance from verified records, never from files found. Preserve rejection
  evidence under `expected_rejections/`, separately from `failed_cases/`.
  Quick requires 40 successful models and 40 audits per representation.
- Report exact commands, report paths, source revision and dirty state,
  environment, runtime and executed/expected coverage.
- Report local and hosted results separately. Quick success is not full
  validation; identify all outstanding required runs.

GitHub validation:
- The Validation workflow runs quick validation on PRs targeting `main` and
  pushes to `main`. Manual runs offer quick/full, defaulting to full.
- There are no scheduled runs. Inspect hosted reports and logs before claiming
  CI validation succeeded.
- Workflow checks do not enforce merge restrictions by themselves. Do not
  change branch protection, rulesets or merge behavior without authorization.

- Do not treat external fixtures as independent validation without overlap checks.
- CLI dry-run is not integration validation and can create directories or inspect
  artifacts. Do not assume it is side-effect-free.

If required validation infrastructure or resources are missing, report the gap,
the unexecuted configurations, and the remaining work. Do not claim full validation.

## Completion report

Report:
- Changes and files modified.
- Scientific/methodological implications and compatibility effects.
- Exact tests/commands actually executed and relevant environment/configuration.
- Model/selection coverage as executed/expected counts, with sampling, model
  variants, data cases, and workflow routes stated separately.
- Passed, failed, skipped, and unsupported configurations, with reasons.
- Reproducibility concerns, known limitations, and outstanding validation.

Distinguish executed results from proposed checks. Never claim tests passed
unless they ran successfully. Mandatory incomplete validation must remain explicit.

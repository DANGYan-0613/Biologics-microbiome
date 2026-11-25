# Biologics Microbiome Prediction

Current analysis: fixed three-group Leave-One-Group-Out cross-validation
(LOGO-CV), public development in 175 patients, followed by evaluation of one
locked microbiome model in the 52-patient Chinese cohort.

**The same three LOGO folds are used for pipeline selection and reporting.
Selected CV performance is an internally optimized development estimate,
not an unbiased nested-CV estimate.** No random patient-level train/validation
split or additional inner CV is introduced.

## Study Groups

| Validation group | Studies | Training N | Validation N |
|---|---|---:|---:|
| G1 | PRJNA685168 | 61 | 114 |
| G2 | SRP057027 | 136 | 39 |
| G3 | PRJNA384246, PRJNA389280, PRJNA398089 | 153 | 22 |

Original metadata `group=1` means Response; `group=0` means Non-response.
The classifier target is **Non-response = 1 - original group**.

## Workflow

For every candidate, run all three folds. Within each fold, fit all raw
prevalence/variance filtering, CLR parameters, standardization, feature
selection and classifier coefficients/trees on the two training groups only.
Apply the fitted pipeline to the completely held-out group.

- Raw prevalence >=0.20; sample variance >1e-12.
- Species: natural-log CLR, half the training minimum positive abundance as
  pseudocount, using **all training-retained Species** as the CLR denominator.
- PWY: original abundance scale; exclude UNMAPPED/UNINTEGRATED summary rows.
- Both blocks: training mean and sample SD (`ddof=1`).
- Joint Species+PWY pool; no fixed per-block quota.
- Selectors: ANOVA-F ranking, nonzero Elastic Net coefficient ranking, joint
  ExtraTrees-RFE. Total counts: 4, 8, 12, 16, 20, 24, 32, 48, 64, plus an
  all-retained comparator. Sparse supports are never padded with zeros.
- Classifiers: 8 ExtraTrees, 4 Random Forest and 3 L2-logistic configurations.
  The full grid contains 420 candidates; 375 were feasible in all three folds.

Prefer mean group AUROC within 0.01 of the maximum, then worst-group AUROC
within 0.02 of the best in that pool. Among eligible candidates prefer fewer
features, greater pairwise-Jaccard stability, higher pooled nonresponse AP,
then the fully specified complexity/deterministic tie-breaks in
`logo_benchmark.SELECTION_RULE`. The complete grid and rule are saved before
the run in `analysis_plan.json`. This is a finite comparison, not a claim of
exhaustive search across every possible algorithm or hyperparameter.

Refit the chosen procedure on all 175 patients. Lock one threshold by maximum
Youden J on its pooled public OOF predictions (highest finite cutoff on ties).
Predict Non-response when `q >= cutoff`. Never choose or change it using
Chinese labels. Save and hash label-free external predictions before the
outcome join; use external outcomes only for evaluation.

## Recorded Result

Selected pipeline **C0319**: joint ExtraTrees-RFE, total K=16, then ExtraTrees
with 700 trees, depth 5, minimum leaf size 2, `max_features="sqrt"`,
`class_weight=None`, `bootstrap=False`, seed 20261125.
The RFE selector is a separate 300-tree estimator (depth 5, leaf 2, sqrt),
eliminating features recursively along the declared count grid.

| Development evaluation | N | Species + PWY | AUROC | Non-response AP |
|---|---:|---:|---:|---:|
| G1 | 114 | 11 + 5 | 0.682266 | 0.726934 |
| G2 | 39 | 11 + 5 | 0.808571 | 0.620707 |
| G3 | 22 | 14 + 2 | 0.676190 | 0.830540 |
| Pooled OOF | 175 | Fold-specific | 0.658699 | 0.675844 |

Mean group AUROC: 0.722343. Full-public refitting selected 13 Species + 3 PWY.
Only 2 Species and 1 PWY were selected in all three development folds;
stability is descriptive and should not be overstated.

Locked public OOF Non-response cutoff: **0.5417599267387728**.

| Locked Chinese evaluation | Value |
|---|---:|
| N / Non-response / Response | 52 / 13 / 39 |
| AUROC | 0.808679 |
| Non-response AUPRC (average precision) | 0.586100 |
| Sensitivity / Specificity | 0.846154 / 0.692308 |
| Accuracy / PPV / NPV / F1 | 0.730769 / 0.478261 / 0.931034 / 0.611111 |
| TP / FP / TN / FN (Non-response positive) | 11 / 12 / 27 / 2 |
| Brier score, raw uncalibrated probabilities | 0.219535 |

AUROC, AP and Brier do not depend on the classification cutoff. Group-specific
classification metrics use the same pooled-public cutoff, not a separately
optimized per-group threshold. Nonresponse AP has an external prevalence
baseline of 0.25; it is not Response AP or trapezoidal PR area.

## Setup and Run

Recorded environment: Python 3.12.3. Create an isolated environment and install
`requirements.txt`. These pins reflect the completed analysis; exact model
hashes may depend on matching Python and dependency versions.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_logo_benchmark.py

python logo_benchmark.py --input-dir data/public --output outputs/development
python verify_logo_results.py --output outputs/development

python validate_external_C0319.py --development-dir outputs/development --input-dir data/external
python verify_external_C0319.py --development-dir outputs/development --input-dir data/external
```

All output directories are new-only: existing results are never overwritten.
There is no need to rerun model development just to inspect or verify saved
results. External validation checks the exact recorded model SHA256 and data
SHA256 values; a changed model or metadata version causes an explicit stop.
Do not bypass these guards to silently substitute outcomes.

## Private Input Contract

Data, per-patient outputs and model binaries are **not included**. Supply the
authorized files locally. Input SHA256 values are fixed in the code to ensure
that the recorded data versions, rather than modified labels, are used.

`data/public/` requires:

```text
public_175_meta_species_pwy_raw_integrated_meta.tsv
public_175_meta_species_pwy_raw_integrated_species.tsv
public_175_meta_species_pwy_raw_integrated_pwy.tsv
```

`data/external/` requires:

```text
yyyy_external_0928_pwy_matched_integrated_species_meta.tsv
yyyy_external_0928_pwy_matched_integrated_species_species.tsv
yyyy_external_0928_pwy_matched_integrated_species_pwy.tsv
```

All are sample-by-feature TSVs with a unique `id` column. Metadata includes
`group`; public metadata also includes `dataset`. Species columns start with
`species__`, PWY columns with `pwy__`. Raw relative-abundance inputs are on the
0-1 scale; no extra /100 or normalization is applied. Clinical fields are not
predictors in this microbiome-only analysis. IDs must match one-to-one; samples
are not silently intersected, dropped, or deduplicated.

The 175-person public analysis intentionally retains three stored whole-zero
PWY profiles under the approved policy; these are explicitly audited encodings,
not established biological absence. No Chinese profile is empty. All 119
public-retained Species and 388 PWY components required at inference were
present externally. Missing components cause an error, not automatic zero filling.
Even Species not selected by the classifier remain necessary for the CLR
denominator. Do not infer using only the final 16 raw abundance columns.

## Code and Outputs

- `logo_benchmark.py`: all 420 candidates, fixed LOGO, OOF selection and final fit.
- `pipeline_core.py`: input auditing, fixed groups and exact preprocessing.
- `elastic_net_support.py`: training-only sparse-selector convergence handling.
- `test_logo_benchmark.py`: synthetic unit tests, including held-out mutation checks.
- `verify_logo_results.py`: independent saved-result recalculation without fitting.
- `validate_external_C0319.py`: locked external inference and evaluation.
- `verify_external_C0319.py`: independent external checks, no fitting or tuning.
- `source_provenance.json`: original script hashes and AST-equivalence checks.

Generated outputs include all candidate scores and OOF predictions,
fold-specific selected Species/PWY identities and frequency stability,
prediction distributions, locked Youden cutoff, complete preprocessing state,
final signature/MDI importance, model registry, exact input snapshots and logs.
Generated files may contain sensitive records and must remain private.
Tree importance is mean decrease in impurity, **not PLS VIP**.

## Interpretation and Scope

This snapshot replaces the previous repository analysis rather than mixing
in older clinical/combined models, plotting scripts or ZIP archives. It contains
only this completed microbiome workflow. No clinical/combined comparator was
fitted in this run. Repository cleanup preserves the Git history.

Within-candidate held-out patients do not fit that candidate's preprocessing,
selector or classifier, but their outcomes do influence pipeline selection
and the pooled-development cutoff. Repeated historical searches may introduce
additional optimism beyond the current grid. The Chinese cohort has already
been examined in earlier project iterations: this evaluation does not reset
that history or make it prospectively untouched. No clinical utility claim is
established by these metrics; this is research code, not clinical decision software.

Publishing and portability checks do not refit the production model. They only
change input/output CLI defaults, remove personal filesystem paths, extract
the unchanged helper functions and verify compatibility with existing results.

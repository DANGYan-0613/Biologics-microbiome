# Species + PWY ExtraTrees Prediction

This repository contains executable code, not only an analysis proposal.
The original requested design is preserved in [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md).

**Status:** the expanded ExtraTrees-only search is implemented and tested, but
has not been run on the real cohorts. The previously completed C0319 workflow
is included separately for reproducibility; its results must not be described
as results of the expanded search. See [docs/C0319_REFERENCE.md](docs/C0319_REFERENCE.md).

## Main Code

| File | Purpose |
|---|---|
| `run_extratrees_logo.py` | New ExtraTrees-only, finite-search development entry point |
| `logo_benchmark.py` | Shared three-fold LOGO engine, preprocessing, selectors, comparison, threshold and final model |
| `pipeline_core.py` | Public input auditing, exact groups and training-only transformation |
| `elastic_net_support.py` | Sparse Elastic Net selector with convergence checks |
| `evaluate_external.py` | Locked-model external evaluation, 95% bootstrap CIs and Brier score |
| `make_figures.py` | Feature-count, parameter, ROC/PR, stability, importance, confusion and calibration figures; Methods text |
| `verify_logo_results.py` | Independent checks on saved development outputs without fitting |
| `validate_external_C0319.py` | Reproduce the already completed, specifically locked C0319 external evaluation |
| `verify_external_C0319.py` | Independent checks for that recorded C0319 evaluation |
| `test_logo_benchmark.py`, `test_extended_pipeline.py` | Leakage, grid, numerical and synthetic end-to-end tests |

## Fixed Design

Public development includes175 patients, without silent sample removal:

| Validation group | Studies | Train N | Validation N |
|---|---|---:|---:|
| G1 | PRJNA685168 | 61 | 114 |
| G2 | SRP057027 | 136 | 39 |
| G3 | PRJNA384246, PRJNA389280, PRJNA398089 | 153 | 22 |

No random80/20 or70/30 split and no extra inner CV are used.
Original `group=0` is Non-response; original `group=1` is Response.
All models here predict **Non-response** as the positive event.

Within each candidate/fold, use only the two training groups for prevalence
>=20% and raw sample variance>1e-12 filtering; natural-log Species CLR using
half the minimum positive training abundance and all training-retained Species
as the CLR reference; original normalized PWY values excluding UNMAPPED and
UNINTEGRATED; training mean and sample SD (`ddof=1`) standardization; feature
selection; and classifier fitting. Apply these operations unchanged to the
held-out group. Species and PWY are one pool, with no fixed per-block quota.

Compare ANOVA-F, nonzero Elastic Net coefficient ranking and ExtraTrees-RFE at
4,8,12,16,20,24,32,48,64 total features, plus a shared all-retained comparator.
Do not pad sparse supports with zero-coefficient features. Training rankings,
RFE elimination-stage importance, selected Species/PWY, fold frequency and
pairwise Jaccard stability are recorded. The RFE selector is a fixed300-tree
depth5/leaf2/sqrt estimator, separate from the final classifier.

## Expanded Search

The declared parameter space has144 possible ExtraTrees configurations:

| Parameter | Values |
|---|---|
| `n_estimators` | 300,700 |
| `max_depth` | 3,5,7 |
| `min_samples_leaf` | 2,4,6 |
| `min_samples_split` | 2,5 |
| `max_features` | sqrt,0.5 |
| `class_weight` | None,balanced |

For efficiency, the script fixes24 base configurations before seeing any data:
8 previously evaluated unweighted base settings, one additional base setting
covering the new parameter levels, and15 seeded draws without replacement from
the remaining base space. Each base is paired with both class-weight options.
Thus48 unique classifier configurations are compared with28 selector/count
options: **1,344 complete candidate pipelines**, each using all3 LOGO folds.
The seed is20261125; criterion is gini, bootstrap is False. This is a finite,
predeclared subset, not an exhaustive search of all144 settings. Algorithmic
randomness/forest sampling is not an additional patient-validation split.

The complete actual candidate manifest is saved before fitting. Prefer mean
group AUROC within.01 of the maximum, then worst-group AUROC within.02 of the
best in that pool, then fewer features, higher mean pairwise Jaccard, pooled
nonresponse AP and the documented complexity/tie rules. Mean-group and pooled
OOF AUROC are distinct outputs. Do not select on Chinese performance.

After selection, refit the same filtering/transformation/selector procedure
on all175 patients and fit one final classifier. The final feature identities
may differ from any fold-specific list. Lock one threshold by pooled-public-OOF
Youden J; choose the highest finite threshold on ties. Classify Non-response
when `q >= threshold`. Do not substitute.5 or optimize on external labels.

**The same LOGO folds are used for selection and reporting. Selected CV
performance is an internally optimized development estimate, NOT an unbiased
nested-CV estimate.** Repeated historical searches may introduce further
optimism. The Chinese cohort was previously explored, not prospectively untouched.

## Run

Recorded runtime is Python3.12.3. Dependencies are pinned in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_logo_benchmark.py test_extended_pipeline.py

# New ExtraTrees-only development; no Chinese file is read.
python run_extratrees_logo.py --input-dir data/public --output outputs/extratrees_v2
python verify_logo_results.py --output outputs/extratrees_v2

# Only after the public model, features and cutoff have been locked:
python evaluate_external.py --development-dir outputs/extratrees_v2 --input-dir data/external

# Tables and PDF/PNG figures; no model fitting or threshold optimization:
python make_figures.py --development-dir outputs/extratrees_v2 --external-dir outputs/extratrees_v2/external --output outputs/figures_v2
```

Output directories are new-only, never overwritten. Evaluation bootstrap
resamples patients to quantify uncertainty of fixed predictions; it does not
fit models, create development splits, change probabilities or tune cutoffs.
Intervals are percentile bootstrap CIs, conditional on the fixed fitted pipeline;
they do not account for development model-selection uncertainty. Undefined
metrics are reported missing with their valid replicate count.

External prediction is frozen and hashed before the outcome join. Only sample
IDs and microbial tables enter inference; age, subtype, biologic, dataset and
response labels are never predictor columns. No probability calibrator is fitted.
The four-quantile-bin calibration plot assesses raw probabilities only.

To reproduce the previous420-candidate C0319 benchmark instead, run
`python logo_benchmark.py --input-dir data/public --output outputs/development`.
That older benchmark also compares RF and logistic candidates; these are NOT
part of the new `run_extratrees_logo.py` classifier search. The C0319-specific
validation scripts intentionally reject a different model SHA256.

## Private Data Contract

The repository contains no patient tables, per-patient predictions or model
binaries. Obtain authorized data separately. Do not commit generated `data/`
or `outputs/`; their snapshots, matrices and predictions can be sensitive.

`data/public/` must contain:

```text
public_175_meta_species_pwy_raw_integrated_meta.tsv
public_175_meta_species_pwy_raw_integrated_species.tsv
public_175_meta_species_pwy_raw_integrated_pwy.tsv
```

`data/external/` must contain:

```text
yyyy_external_0928_pwy_matched_integrated_species_meta.tsv
yyyy_external_0928_pwy_matched_integrated_species_species.tsv
yyyy_external_0928_pwy_matched_integrated_species_pwy.tsv
```

These are sample-by-feature TSVs with unique `id`. Public metadata contains
`dataset` and `group`; external metadata contains `group`. Species/PWY names
start with `species__`/`pwy__`. All ID sets must match one-to-one.
Raw abundances already use the0-1 relative-abundance scale; no extra /100 is
applied. SHA256 guards freeze the exact approved input and outcome versions.
Changed files cause an explicit stop, not silent substitution.

All175 public patients remain included under the approved policy, including
3 stored whole-zero PWY profiles explicitly flagged in the audit. These zeros
are not established biological pathway absence. No external patient is dropped.
All training-retained Species required for the CLR denominator must be present
at inference, not just classifier-selected Species. Missing retained features
raise an error rather than being automatically imputed as zero.

## Generated Outputs

Development saves complete candidate/configuration tables,175-patient OOF
predictions for every valid pipeline, group-specific performance/probability
distributions, fold-level selected Species/PWY identities, selector rankings,
feature stability, the locked threshold, final feature list/MDI importance,
fitted model+preprocessing, input snapshots, software versions and run logs.

The plotting entry point generates feature-count vs mean LOGO AUROC,
hyperparameter comparison, group-specific ROC and PR curves, feature-stability
heatmap and final-importance plot. External outputs include ROC/PR, confusion
matrix, calibration plot, Brier, performance and95% intervals. MDI is NOT PLS VIP.
The feature-count curves maximize across development configurations and are
explicitly labeled development-optimized. Error bars are group SD, not CIs.
`MANUSCRIPT_METHODS.txt` describes the actual completed run's selected settings.

This is research software, not clinical decision software. Existing C0319
performance does not establish that the expanded, currently unrun search is better.

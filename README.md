# Study-Grouped Microbiome Response Prediction

Portable code for the locked version 47/48 workflow. This is a code-only
reproduction package, not a newly optimized model or a new analysis result.
It does not contain patient data, identifiers, fitted models, or credentials.

## Scientific Scope

- Positive outcome: `group=1` responder; `group=0` nonresponder.
- Disease subtype: `subtype=1` UC; `subtype=2` CD.
- Clinical predictors: subtype and biologic; adult is not used.
- Fixed groups: G1=PRJNA685168; G2=SRP057027;
  G3=PRJNA384246 + PRJNA389280 + PRJNA398089.
- Each group is held out once; no original study is split across train/test.
- Public pipeline selection uses microbial predictors only. Clinical and
  Combined OOF predictions supplement fixed-model comparisons and thresholds;
  they do not select the microbial pipeline.
- Every training split learns prevalence filtering, pseudocount, transformation,
  scaling, association ranking, and clinical categorical levels independently.
- The same public held-out groups historically selected configurations.
  These results are **grouped-CV development estimates, not unbiased strict
  nested-CV validation**. Prior external inspection and metadata changes also
  prevent describing the historical external data as untouched confirmatory
  validation.

## Locked Pipeline

`configs/locked_v48.json` records Species CLR, PWY original, prevalence >=0.20,
nonzero raw variance >1e-12, 16 Species + 6 PWY, and ExtraTrees with 700 trees,
depth 5, minimum leaf size 2, sqrt feature sampling, balanced class weights.

The training-only pseudocount is half the smallest positive retained abundance.
Species CLR is computed across **all prevalence-retained Species before selecting
16**, not across just the final 16. Scaling uses training means and sample SD
(ddof=1). Feature ranking uses standardized within-study responder/nonresponder
mean differences: absolute weighted mean effect times sign consistency divided
by one plus between-study effect SD. This is not VIP, RF-RFE, or impurity ranking.

The 16+6 subset sizes were selected from a prespecified candidate grid; they are
not an unconstrained automatically discovered number. Each fold has its own
training-selected identities; final identities are selected again using all
175 public patients. Combined uses the **same microbial features** and
ExtraTrees configuration, with subtype/biologic dummy columns added. It is not
stacking. Clinical-only is fixed L2 logistic regression (C=1, liblinear,
balanced class weights).

## Installation

Tested with Python 3.12 and the exact dependencies in `requirements.txt`.
Reproduction across different sklearn versions is not guaranteed.
CSV probabilities are read with round-trip floating-point parsing. Parallel
tree prediction can differ at approximately 1e-16 due to summation order;
numeric reproduction checks therefore use an absolute tolerance of 1e-12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Input Tables

All abundance matrices are **sample-by-feature TSVs**, with an `id` column.
Metadata and both matrices must have identical unique IDs. No sample is
silently dropped or intersected. Abundances must be finite, nonnegative, and
have compatible definitions/units across cohorts.

Public metadata requires `id`, `dataset`, `group`; final fitting and clinical
comparison additionally require `subtype`, `biologic`. Prediction reads only
`id`, `subtype`, `biologic`, and never reads `group` or `dataset`. Evaluation
reads the external outcome metadata after prediction files exist.

Use the frozen study inputs that produced the original results, not modified
live metadata. Data are not provided here and must be obtained under appropriate
permissions. Do not upload individual-level clinical data or fitted models
without verifying consent, authorization, and repository access settings.

## Commands

All commands default to **check-only** unless `--run` is supplied. Existing
output directories are never overwritten. Paths below are placeholders.

### Reproduce the Locked Pipeline Without Repeating Model Selection

```bash
python prediction_pipeline.py public-oof --run \
  --meta data/public_meta.tsv --species data/public_species.tsv \
  --pwy data/public_pwy.tsv --config configs/locked_v48.json \
  --output results/public_oof

python prediction_pipeline.py fit-final --run \
  --meta data/public_meta.tsv --species data/public_species.tsv \
  --pwy data/public_pwy.tsv --config results/public_oof/config.json \
  --cutoffs results/public_oof/public_oof_thresholds.json \
  --output results/final_models

python prediction_pipeline.py predict --run \
  --meta data/external_meta.tsv --species data/external_species.tsv \
  --pwy data/external_pwy.tsv --models results/final_models \
  --cutoffs results/final_models/public_oof_thresholds.json \
  --trust-model-files --output results/external_predictions

python prediction_pipeline.py evaluate --run \
  --predictions results/external_predictions/external_predictions.csv \
  --meta data/external_meta.tsv \
  --cutoffs results/final_models/public_oof_thresholds.json \
  --output results/external_evaluation
```

Each model's cutoff maximizes pooled public OOF Youden J. Predictions >= cutoff
are responders. Neither cutoff=0.5 nor a Chinese-optimized cutoff is used.
Final fitting verifies that the OOF configuration and public input hashes match.

Prediction can instead read the original trusted version-48 complete pipeline
pickles by setting `--models` to that model directory and `--cutoffs` to its
three-model public OOF threshold registry. It verifies each pickle SHA256.
**Pickles can execute code**: never use `--trust-model-files` with untrusted files.

Missing training-retained microbial columns fail by default. Only after confirming
complete compatible taxonomic/functional profiles, use
`--allow-missing-microbial-columns` to reproduce historical raw-zero filling.
Raw zero filling occurs before CLR/scaling and does not establish biological
absence. It is logged for all retained columns, not just selected ones.
Test-only clinical levels retain historical all-zero reference encoding and are
reported; they are not learned from held-out patients.

### Optional Historical Public Development Reproduction

This is computationally expensive and **not needed for inference**.

```bash
python prediction_pipeline.py screen --run \
  --meta data/public_meta.tsv --species data/public_species.tsv \
  --pwy data/public_pwy.tsv --output results/screen

python prediction_pipeline.py tune --run \
  --meta data/public_meta.tsv --species data/public_species.tsv \
  --pwy data/public_pwy.tsv --screen-config results/screen/selected_config.json \
  --output results/tune
```

Stage 1 compares 336 transform/count/classifier candidates. Historical linear
SVM probability estimation uses sklearn's internal probability calibration;
that internal patient-level procedure is another reason not to label this
screen strict study-aware nested CV. The final ExtraTrees pipeline does not use
that procedure or any fitted probability calibrator.

Stage 2 fixes the selected transforms, counts, and ExtraTrees family and compares
108 parameter configurations. If stage 1 selects another classifier, stage 2
fails explicitly rather than silently forcing ExtraTrees.

Both stages use: mean-fold AUROC within 0.01 of best; min-fold AUROC within
0.03 of the best minimum in that pool; another within-0.01 mean-AUROC filter;
then fewer features, higher mean-fold AUROC, higher pooled AUROC. AP is reported
but is not a selection criterion. No external input is accepted in these stages.

To use a new screened/tuned configuration, pass its selected_config.json to
public-oof and then the resulting config.json to fit-final. Such a run is a new
development result and must not be selected based on external outcomes.

## Evaluation Outputs

- AUROC and Average Precision (AP, the reported AUPRC), overall and UC/CD/drug strata.
- Overall external ROC and precision-recall plots for the three frozen models.
- TP/FP/TN/FN and sensitivity, specificity, accuracy, PPV, NPV, F1 at locked cutoffs.
- Two-sided 95% Wilson intervals for the five proportions; F1 Wilson indirect interval.
- Paired response-stratified bootstrap AUROC intervals (10000 overall, 5000 subgroup).
- Brier score and paired response-stratified bootstrap CI (10000 resamples).
- Microbiome reliability diagrams with 4 and 5 quantile bins and Wilson bin intervals.
- Diagnostic CITL intercept with slope fixed at 1 and a profile-likelihood CI.
  **The intercept is never applied to predictions.** No calibrator is fitted
  for deployment. Raw ExtraTrees predict_proba remains the basis of all scores.
- Audit JSONs, hashes, clinical-level warnings, feature availability and stability tables.

Single-class strata have no AUROC/AP or specificity if no nonresponders; undefined
denominators produce NA. A high value from a tiny subgroup is not independent
validation. CI estimates condition on the frozen model and observed samples,
not on model-selection uncertainty. Subtype/treatment strata can overlap.

The original micro-model reference values are mean-fold public AUROC 0.713956,
pooled public AUROC 0.641850, and cutoff 0.46610918702271736. In the archived
52-patient external cohort: AUROC 0.777120, AP 0.914850, Brier 0.252404 and
CITL 1.345409. These describe the historical inputs, not performance guarantees.

## References

- https://scikit-learn.org/stable/modules/calibration.html
- https://arxiv.org/abs/2309.14621 (F1 Wilson indirect interval)
- https://doi.org/10.1186/s12916-019-1466-7 (calibration assessment)

No license is imposed by this package; the repository owner should choose an
appropriate license before public distribution.

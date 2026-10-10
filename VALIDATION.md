# Packaging Verification

Checked on 2026-10-10 with Python 3.12 and the pinned dependencies.

## Automated Tests

Six unittest tests passed, covering training-only preprocessing, raw-zero
filling opt-in, duplicate-header rejection, candidate selection, Wilson/F1/CITL
methods, CLI dry-run behavior, complete synthetic OOF/final-fit/predict/evaluate
execution, outcome-label invariance in prediction mode, and overwrite refusal.
All classifier fitting in these tests used generated synthetic data only.

## Read-Only Historical Reference Checks

No real-data classifier was fitted or refitted during packaging.

- Final 16 Species + 6 PWY identities matched the archived version-48 list.
- Each of the three fold-specific selected-feature lists matched version 47.
- Retained features, pseudocounts, transformation, means and SDs matched the
  original final preprocessing state exactly.
- Existing trusted Clinical/Microbiome/Combined model predictions for 52 frozen
  external patients were reproduced, with maximum probability errors 0,
  1.11e-16 and 2.22e-16, respectively.
- The six microbiome scope tables reproduced original point estimates,
  AUROC bootstrap intervals, all six classification-metric intervals,
  Brier scores and Brier intervals within absolute tolerance 1e-12.
- Overall microbial CITL and its profile-likelihood interval reproduced the
  original result within 1e-12.
- Calibration plot bin means/counts matched sklearn; figure layouts were
  visually checked in the local packaging QA run.
- No original fitted-model files were modified. No patient-level data,
  prediction files, metadata, model pickles or local absolute paths are
  included in the distribution.

## Not Claimed

The complete 336-candidate and 108-candidate real-data development searches
were not rerun during packaging. The underlying transformations, ranking,
candidate definitions, selection rule and seeds were reproduced in portable
code, but a fresh run can depend on inputs and software versions.

This package does not establish a new performance improvement, undo historical
model-selection reuse, or provide untouched independent external validation.
Do not select a model or cutoff after comparing external outcomes.

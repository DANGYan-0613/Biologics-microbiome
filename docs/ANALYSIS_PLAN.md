# Development of a Species + PWY-Based ExtraTrees Prediction Model

## 1. Research Objective

Develop a reproducible machine-learning pipeline to predict biologic treatment non-response in inflammatory bowel disease using pretreatment gut metagenomic profiles.

The pipeline consists of two core components:

1. **Feature selection:** Identify predictive features from the integrated Species + PWY matrix.
2. **ExtraTrees modeling:** Optimize ExtraTreesClassifier hyperparameters and construct the final prediction model.

Optimize the complete pipeline using study-grouped cross-validation.

## 2. Study Population and Cross-Validation

The development dataset includes 175 patients from five public studies, organized into three predefined groups:

| Fold | Training groups | Validation group |
|---|---|---|
| 1 | G2 + G3 (n=61) | G1 (n=114) |
| 2 | G1 + G3 (n=136) | G2 (n=39) |
| 3 | G1 + G2 (n=153) | G3 (n=22) |

Use fixed Leave-One-Group-Out Cross-Validation (LOGO-CV).

Define treatment non-response as the positive class.

## 3. Data Preprocessing

Use the integrated Species + PWY abundance matrices.

For each LOGO fold:

1. Apply prevalence filtering (≥20%) and low-variance filtering using training data.
2. Perform natural-log CLR transformation of Species abundances, using a training-derived pseudocount and all training-retained species as the CLR reference set.
3. Process PWY abundances on their original normalized scale, excluding UNMAPPED and UNINTEGRATED summary features.
4. Standardize each feature using training-set mean and standard deviation.
5. Combine Species and PWY into one feature matrix.

Apply the fitted preprocessing pipeline to the held-out validation group.

Maintain consistent sample IDs, feature identities, and study assignments.

## 4. Part I — Feature Selection

Evaluate three feature selection methods:

1. **ANOVA-F ranking**
2. **ExtraTrees-based Recursive Feature Elimination (RFE)**
3. **Elastic Net-based selection**

Evaluate the following candidate feature counts:

**4, 8, 12, 16, 20, 24, 32, 48, 64, and all eligible features.**

Select features from the integrated Species + PWY matrix without a predefined ratio between the two feature types.

Perform feature selection independently within each LOGO training fold.

Record:

- Selected feature names.
- Number of selected Species and PWY.
- Feature ranking or importance.
- Selection frequencies across folds.
- Pairwise Jaccard stability.

## 5. Part II — ExtraTrees Model Development

Use **ExtraTreesClassifier** as the prediction algorithm.

Optimize the following hyperparameters using a predefined finite search space:

| Parameter | Candidate values |
|---|---|
| n_estimators | 300, 700 |
| max_depth | 3, 5, 7 |
| min_samples_leaf | 2, 4, 6 |
| min_samples_split | 2, 5 |
| max_features | sqrt, 0.5 |
| class_weight | None, balanced |

Use:

- `criterion="gini"`
- `bootstrap=False`
- `random_state=20261125`

Use a computationally efficient search strategy to evaluate a manageable number of complete pipeline configurations.

For each configuration, perform all three LOGO folds, including training-specific preprocessing, feature selection, ExtraTrees fitting, and validation prediction.

## 6. Model Evaluation and Selection

Calculate the following development-stage metrics for each candidate pipeline:

**Primary metric**

- Mean AUROC across G1, G2, and G3.

**Secondary metrics**

- Worst-group AUROC.
- Pooled out-of-fold AUROC.
- Non-response average precision (AUPRC).
- Feature selection stability.
- Total number of selected features.
- Model complexity.

Select the final pipeline using the following hierarchical criteria:

1. Identify candidates with mean group AUROC within 0.01 of the highest observed value.
2. Within this subset, prioritize candidates with worst-group AUROC within 0.02 of the best.
3. Prefer fewer selected features.
4. Prefer higher feature selection stability.
5. Use pooled non-response AUPRC and model complexity as additional tie-breakers.

Apply the same predefined selection criteria to every candidate.

Report the selected LOGO-CV performance as an internally optimized development estimate.

## 7. Final Model Construction

After selecting the optimal feature selection method, feature count, and ExtraTrees hyperparameters:

1. Refit preprocessing using all 175 development patients.
2. Apply the selected feature selection procedure.
3. Record the final Species and PWY features.
4. Train ExtraTrees using all 175 patients.
5. Save the complete fitted pipeline, including preprocessing, selected features, classifier, and configuration.

## 8. Classification Threshold

Use pooled development out-of-fold predicted non-response probabilities to determine one classification threshold using Youden's index:

**J = Sensitivity + Specificity − 1**

Select the threshold maximizing Youden's J.

Apply the locked threshold to the final model's external predictions.

## 9. External Evaluation

Evaluate the finalized pipeline using the 52-patient Chinese cohort.

Generate predicted non-response probabilities and calculate:

- AUROC with 95% confidence interval.
- Non-response AUPRC.
- Sensitivity and specificity.
- PPV and NPV.
- Accuracy and F1-score.
- Confusion matrix.
- Brier score and calibration assessment.

Preserve the distinction between the previously explored external cohort and a prospectively untouched confirmatory validation cohort.

## 10. Required Outputs

Produce the following:

1. **Model comparison table:** All evaluated feature selection methods, feature counts, ExtraTrees configurations, and development performance.
2. **Feature selection performance plot:** Number of selected features versus mean LOGO AUROC.
3. **Hyperparameter comparison:** Performance of tested ExtraTrees configurations.
4. **Group-specific ROC and PR curves:** G1, G2, and G3.
5. **Feature stability heatmap:** Selection frequency and consistency across folds.
6. **Final feature list:** Selected Species and PWY names, importance scores, and feature types.
7. **Final model configuration:** Selected feature selection method, feature count, and ExtraTrees hyperparameters.
8. **External evaluation:** ROC curve, PR curve, confusion matrix, calibration plot, and performance metrics.
9. **Reproducible code and saved pipeline:** Including fitted transformations, model, threshold, and software versions.
10. **Manuscript-ready Methods:** Describe preprocessing, feature selection, ExtraTrees optimization, LOGO-CV, model selection, and external evaluation.

## Final Objective

Identify the most robust and reproducible **Species + PWY feature selection strategy and ExtraTrees model configuration**, maximizing cross-study predictive performance while maintaining feature stability, model simplicity, and methodological reproducibility.
# Model comparison discussion

## Results summary
- The highest available F1 score is achieved by **Decision Tree** (F1 = 0.684160552438498).
- Additional metrics (precision, recall, balanced accuracy, ROC-AUC, PR-AUC, log loss) support the final selection where available.

## Observations
- F1 is used as the primary ranking metric because it balances precision and recall under class imbalance.
- Probability-based metrics (ROC-AUC, PR-AUC, log loss) are included only when probability outputs are available.

## Limitations
- Some models may be missing or evaluated from saved metrics only, which may reflect different splits if pipelines are not available.
- Feature engineering is limited to simple derived variables and one-hot encoding; interaction effects may not be fully captured by linear models.
- High-cardinality categoricals can increase dimensionality and may require alternative encodings or category grouping for additional stability.

## Improvements and future work
- Add probability calibration (e.g., Platt scaling or isotonic calibration) and evaluate calibration curves.
- Use a consistent saved split index file (train/test indices) to ensure identical evaluation across notebooks and environments.
- Expand hyperparameter search using randomized search with time/compute budgets and cross-validated reporting.
- Investigate feature grouping for rare categories and perform ablation studies to measure feature engineering impact.
- Evaluate robustness across multiple random seeds and report confidence intervals for key metrics.

Logistic Regression (baseline) — report notes

Modeling decisions:
- Leakage columns removed before training.
- Missing values imputed; categoricals one-hot encoded; numeric features scaled.
- Regularization tuned using stratified cross-validation with F1 selection.

Findings to report:
- CV results: include best mean F1 and parameter settings.
- Test metrics: include F1, precision, recall, ROC-AUC, PR-AUC (if computed).
- Threshold analysis: report whether a threshold other than 0.50 improves the target metric.
- Interpretability: discuss the most influential positive/negative coefficients.
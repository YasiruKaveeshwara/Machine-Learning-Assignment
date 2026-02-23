# Missing artifacts report

## Expected paths that are missing
- Missing: `artifacts\models\knn_pipeline.joblib`
- Missing: `artifacts\metrics\knn_test_metrics.json`
- Missing: `artifacts\metrics\knn_best_params.json`
- Missing: `artifacts\metrics\knn_cv_results.csv`
- Missing: `artifacts\models\rf_pipeline.joblib`
- Missing: `artifacts\metrics\rf_test_metrics.json`
- Missing: `artifacts\metrics\rf_best_params.json`
- Missing: `artifacts\metrics\rf_cv_results.csv`

## Files found under artifacts/ (possible misplacements)
- test_metrics.json:
  - `artifacts\metrics\dt_test_metrics.json`
  - `artifacts\metrics\logreg_test_metrics.json`
- pipeline.joblib:
  - `artifacts\models\dt_pipeline.joblib`
  - `artifacts\models\logreg_pipeline.joblib`
- best_params.json:
  - `artifacts\metrics\dt_best_params.json`
  - `artifacts\metrics\logreg_best_params.json`
- cv_results.csv:
  - `artifacts\metrics\dt_cv_results.csv`
  - `artifacts\metrics\logreg_cv_results.csv`
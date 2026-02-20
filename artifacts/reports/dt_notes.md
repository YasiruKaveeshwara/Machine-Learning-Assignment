# Decision Tree Model Notes

## Model Overview
- **Algorithm**: Decision Tree Classifier (CART)
- **Task**: Binary classification (hotel booking cancellation prediction)
- **Target Variable**: `is_canceled`

## Data
- **Dataset**: `hotel_bookings_dedup.csv`
- **Total samples**: 87,396
- **Train samples**: 69,916
- **Test samples**: 17,480
- **Test size**: 20%
- **Stratified split**: Yes

## Preprocessing
- **Leakage columns dropped**: reservation_status, reservation_status_date
- **Categorical encoding**: One-hot (min_frequency=0.01)
- **Numeric scaling**: StandardScaler (applied via shared pipeline)
- **Note**: Decision Trees don't require scaling, but using unified pipeline for consistency

## Hyperparameter Tuning
- **Method**: GridSearchCV with 5-fold stratified cross-validation
- **Scoring metric**: F1 score
- **Parameters tuned**:
  - `max_depth`: [None, 5, 10, 20]
  - `min_samples_split`: [2, 10, 30]
  - `min_samples_leaf`: [1, 5, 10]
  - `class_weight`: [None, 'balanced']

## Best Hyperparameters
```json
{
  "model__class_weight": "balanced",
  "model__max_depth": 20,
  "model__min_samples_leaf": 10,
  "model__min_samples_split": 30
}
```

## Cross-Validation Results
- **Best CV F1 Score**: 0.6843

## Test Set Performance
| Metric            | Value              |
|-------------------|--------------------|
| Accuracy          | 0.7907             |
| Balanced Accuracy | 0.8013             |
| Precision         | 0.5845             |
| Recall            | 0.8248             |
| F1 Score          | 0.6842             |
| ROC-AUC           | 0.8751447764594328 |
| PR-AUC            | 0.7281464072627848 |
| Log Loss          | 1.0364687734010156 |

## Key Observations
1. **Tree Depth**: The optimal `max_depth` balances model complexity with generalization
2. **Class Imbalance**: `class_weight='balanced'` may improve recall for minority class
3. **Interpretability**: Feature importance provides clear insight into predictive factors

## Top 5 Important Features
| feature    |   importance |
|:-----------|-------------:|
| feature_52 |    0.20363   |
| feature_0  |    0.163997  |
| feature_15 |    0.0842606 |
| feature_16 |    0.0838816 |
| feature_59 |    0.0722017 |

## Artifacts Generated
- Model: `artifacts/models/dt_pipeline.joblib`
- Metrics: `artifacts/metrics/dt_*.json/csv`
- Plots: `artifacts/plots/dt_*.png`
- Reports: `artifacts/reports/dt_*.txt/md`

## Notes
- Random state: 42 (for reproducibility)
- Run date: 2026-02-20 22:26:08

Preprocessing summary
- Target column: is_canceled
- Leakage columns removed: reservation_status, reservation_status_date
- ID-like columns treated as categorical: agent, company
- Numeric processing: median imputation, quantile clipping (1%–99%), standard scaling.
- Categorical processing: most-frequent imputation, one-hot encoding with unseen-category handling.
- Two preprocessors saved: sparse (general use) and dense (KNN-friendly).
- Fit performed on the training split only; transformations applied to both train and test after fitting.
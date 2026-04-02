# Machine Learning Assignment — Hotel Booking Cancellation Prediction

## 1. Project Summary

This project predicts whether a hotel booking will be canceled (`is_canceled`) using supervised machine learning and a reproducible notebook workflow.

The final pipeline includes:

- Data setup and cleaning
- Exploratory data analysis (EDA)
- Leakage-safe preprocessing
- Four models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
- Unified final comparison and recommendation

All outputs are written into `artifacts/` and are intentionally overwritten on each run to keep the latest results.

## 2. Repository Layout

```text
Machine-Learning-Assignment/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 00_setup_data.ipynb
│   ├── 01_eda_dataset_understanding.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_logreg.ipynb
│   ├── 04_model_knn.ipynb
│   ├── 05_model_decision_tree.ipynb
│   ├── 06_model_random_forest.ipynb
│   └── 07_model_comparison.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_eval.py
│   ├── metrics.py
│   ├── plots.py
│   └── io_utils.py
├── artifacts/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── reports/
├── requirements.txt
└── README.md
```

## 3. Environment Setup

### 3.1 Open in VS Code

Open the repository root folder (the one containing `notebooks/`, `src/`, `data/`, and `artifacts/`).

### 3.2 Create and activate virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3.3 Install dependencies

```powershell
pip install -U pip
pip install -r requirements.txt
```

### 3.4 Select Python interpreter in VS Code

- Command Palette: `Python: Select Interpreter`
- Select `.venv\Scripts\python.exe`

### 3.5 Register Jupyter kernel

```powershell
python -m ipykernel install --user --name ml-assignment-venv --display-name "ML Assignment (.venv)"
```

### 3.6 Dataset location

Place the dataset at:

- `data/raw/hotel_bookings.csv`

The pipeline will prefer the processed file when available:

- `data/processed/hotel_bookings_dedup.csv`

## 4. End-to-End Notebook Execution Order

Run notebooks in this order:

1. `00_setup_data.ipynb`
2. `01_eda_dataset_understanding.ipynb`
3. `02_preprocessing_pipeline.ipynb`
4. `03_model_logreg.ipynb`
5. `04_model_knn.ipynb`
6. `05_model_decision_tree.ipynb`
7. `06_model_random_forest.ipynb`
8. `07_model_comparison.ipynb`

## 5. Notebook Responsibilities and Main Outputs

### 5.1 `00_setup_data.ipynb`

Purpose:

- Initial dataset loading and sanity checks
- Optional de-duplication baseline creation
- Basic summaries saved for downstream notebooks

Typical outputs:

- `artifacts/data/summary.json`
- `artifacts/data/df_head.csv`
- `artifacts/data/missing_top20.csv`
- `artifacts/data/target_distribution.csv`
- `data/processed/hotel_bookings_dedup.csv`

### 5.2 `01_eda_dataset_understanding.ipynb`

Purpose:

- Understand missing values, distributions, and group-wise cancellation behavior

Outputs:

- `artifacts/data/target_by_group_*.csv`
- `artifacts/data/numeric_describe.csv`
- `artifacts/data/correlation_numeric.csv`
- EDA plots in `artifacts/plots/`

### 5.3 `02_preprocessing_pipeline.ipynb`

Purpose:

- Build leakage-safe preprocessing pipelines
- Save reusable sparse/dense preprocessors

Outputs:

- `artifacts/preprocessing/preprocessor_sparse.joblib`
- `artifacts/preprocessing/preprocessor_dense.joblib`
- `artifacts/preprocessing/preprocess_options_sparse.json`
- `artifacts/preprocessing/preprocess_options_dense.json`
- `artifacts/preprocessing/transform_info_sparse.json`
- `artifacts/preprocessing/transform_info_dense.json`
- `artifacts/preprocessing/feature_names.csv`
- `artifacts/data/train_test_split.json`

### 5.4 `03_model_logreg.ipynb`

Purpose:

- Train/tune Logistic Regression and export full evaluation artifacts

Outputs:

- `artifacts/models/logreg_pipeline.joblib`
- `artifacts/metrics/logreg_best_params.json`
- `artifacts/metrics/logreg_cv_results.csv`
- `artifacts/metrics/logreg_test_metrics.json`
- `artifacts/metrics/logreg_threshold_metrics.csv`
- `artifacts/metrics/logreg_coefficients_top.csv`
- related `logreg_*.png` plots

### 5.5 `04_model_knn.ipynb`

Purpose:

- Train/tune KNN using dense preprocessing
- Evaluate threshold behavior and permutation importance

Outputs:

- `artifacts/models/knn_pipeline.joblib`
- `artifacts/metrics/knn_best_params.json`
- `artifacts/metrics/knn_cv_results.csv`
- `artifacts/metrics/knn_test_metrics.json`
- `artifacts/metrics/knn_threshold_metrics.csv`
- `artifacts/metrics/knn_feature_importance.csv`
- related `knn_*.png` plots

### 5.6 `05_model_decision_tree.ipynb`

Purpose:

- Train/tune Decision Tree and export performance + interpretability artifacts

Outputs:

- `artifacts/models/dt_pipeline.joblib`
- `artifacts/metrics/dt_best_params.json`
- `artifacts/metrics/dt_cv_results.csv`
- `artifacts/metrics/dt_test_metrics.json`
- `artifacts/metrics/dt_threshold_metrics.csv`
- `artifacts/metrics/dt_feature_importance.csv`
- related `dt_*.png` plots

### 5.7 `06_model_random_forest.ipynb`

Purpose:

- Train/tune Random Forest and export evaluation artifacts

Outputs:

- `artifacts/models/rf_pipeline.joblib`
- `artifacts/metrics/rf_best_params.json`
- `artifacts/metrics/rf_cv_results.csv`
- `artifacts/metrics/rf_test_metrics.json`
- `artifacts/metrics/rf_threshold_metrics.csv`
- `artifacts/metrics/rf_feature_importance.csv`
- `artifacts/metrics/rf_confusion_matrix.csv`
- related `rf_*.png` plots

### 5.8 `07_model_comparison.ipynb`

Purpose:

- Evaluate all available models on a consistent split when pipelines are available
- Generate final ranking and recommendation
- Save report-ready comparison plots and notes

Outputs:

- `artifacts/metrics/model_comparison.csv`
- `artifacts/metrics/model_ranking.json`
- `artifacts/metrics/final_recommendation.json`
- `artifacts/reports/model_comparison_notes.md`
- `artifacts/reports/missing_artifacts_report.md`
- `artifacts/plots/compare_f1_bar.png`
- `artifacts/plots/compare_precision_recall_bar.png`
- `artifacts/plots/compare_balanced_accuracy_bar.png`
- `artifacts/plots/compare_roc_overlay.png` (if probabilities available)
- `artifacts/plots/compare_pr_overlay.png` (if probabilities available)

## 6. Reproducibility and Fairness Rules

- Splits are stratified for class balance.
- Random seeds are fixed via `src/config.py`.
- Preprocessing is fitted on training data only.
- Comparison notebook uses one deterministic split for fair model evaluation when pipelines are loadable.
- Artifacts are overwritten each run to avoid stale result confusion.

## 7. Common Troubleshooting

### 7.1 `ModuleNotFoundError: src`

- Open the repository root in VS Code, then rerun from top cells.
- Notebook bootstrap cells auto-detect the repo root.

### 7.2 `IProgress not found` with `tqdm`

- Current notebooks use `tqdm.auto`, which avoids hard dependency on notebook widgets.
- If needed: `pip install ipywidgets`.

### 7.3 Pipeline loading errors (scikit-learn version mismatch)

- If a saved `*.joblib` fails to predict after load, rerun the relevant model notebook to regenerate artifacts in the current environment.

## 8. Team Contributions (Declaration)

| Member   | Contributions                                                                    |
| -------- | -------------------------------------------------------------------------------- |
| Yasiru   | Setup, EDA, preprocessing pipeline, Logistic Regression, model comparison script |
| Member 2 | KNN implementation, video editing                                                |
| Dilhara  | Decision Tree implementation, report formatting                                  |
| Member 4 | Random Forest implementation, GitHub repository management                       |

## 9. Git LFS for Large Model Files

Model files may exceed GitHub's regular file-size threshold. Use Git LFS for `artifacts/models/*.joblib`.

```powershell
git lfs install
git add .gitattributes
git add artifacts/models/*.joblib
git commit -m "Track model artifacts with Git LFS"
```

If large binaries were committed before LFS tracking, coordinate with the team before history rewrite:

```powershell
git lfs migrate import --include="artifacts/models/*.joblib" --everything
```

# Machine Learning Assignment — Hotel Booking Demand (Cancellation Prediction)

## Project Overview

This project builds and compares multiple supervised machine learning models to predict whether a hotel booking will be canceled (`is_canceled`).
The workflow is implemented as a sequence of notebooks supported by reusable Python modules under `src/`.

Primary goals:

- Maintain a leakage-safe ML pipeline (training-only fitting for preprocessing)
- Produce report-ready EDA and evaluation outputs
- Save outputs from every stage into a fixed `artifacts/` structure (overwritten on each run)
- Enable fair model comparison using consistent splitting and preprocessing policies

---

## Repository Structure

```
ML-assignment/
	data/
		raw/
		processed/

	notebooks/
		00_setup_data.ipynb
		01_eda_dataset_understanding.ipynb
		02_preprocessing_pipeline.ipynb
		03_model_logreg.ipynb
		04_model_knn.ipynb
		05_model_decision_tree.ipynb
		06_model_random_forest.ipynb
		07_model_comparison.ipynb

	src/
		__init__.py
		config.py
		data_loader.py
		preprocessing.py
		train_eval.py
		metrics.py
		plots.py
		io_utils.py

	artifacts/
		data/
		preprocessing/
		models/
		metrics/
		plots/
		reports/
```

`artifacts/` contains generated outputs (tables, models, metrics, plots, and report notes).  
Files in `artifacts/` are overwritten each time a notebook is run.

---

## Setup Instructions (VS Code)

### 1) Open the project folder

Open the repository root in VS Code (the folder that contains `src/`, `notebooks/`, and `data/`).

### 2) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3) Install dependencies

`requirements.txt` is already included. Install from it:

```powershell
pip install -U pip
pip install -r requirements.txt
```

### 4) Select the interpreter in VS Code

Ctrl + Shift + P → Python: Select Interpreter

Select: .venv\Scripts\python.exe

### 5) Register the kernel for notebooks

```powershell
python -m ipykernel install --user --name ml-assignment-venv --display-name "ML Assignment (.venv)"
```

### 6) Dataset placement

Place the dataset CSV at:

`data/raw/hotel_bookings.csv`

Optional (recommended for consistent teamwork):

Notebook 00_setup_data.ipynb saves a deduplicated dataset to:

`data/processed/hotel_bookings_dedup.csv`

## Execution Order (Notebooks)

### 00_setup_data.ipynb — Setup and Data Loading

Purpose:

- Load the dataset from `data/raw/`
- Perform safe cleaning (drop exact duplicates)
- Save initial summaries and snapshots to `artifacts/data/`
- Save a deduplicated dataset to `data/processed/` for consistent downstream runs

Key outputs:

- `artifacts/data/summary.json`
- `artifacts/data/df_head.csv`
- `artifacts/data/missing_top20.csv`
- `artifacts/data/target_distribution.csv`
- `data/processed/hotel_bookings_dedup.csv` (optional shared baseline)

### 01_eda_dataset_understanding.ipynb — Exploratory Data Analysis

Purpose:

- Validate data quality and missingness patterns
- Analyze target distribution and cancellation patterns
- Generate report-ready plots and tables

Key outputs:

- Tables in `artifacts/data/` (missingness, describes, grouped cancellation rates)
- Figures in `artifacts/plots/` (target distribution, missingness, histograms, grouped rates, correlation heatmap)
- Notes in `artifacts/reports/eda_insights.md` (report-ready bullet points)

### 02_preprocessing_pipeline.ipynb — Leakage-Safe Preprocessing

Purpose:

- Build reusable preprocessing pipelines
- Fit preprocessing only on the training split
- Save preprocessors for use in all model notebooks

Two preprocessors are saved:

- `artifacts/preprocessing/preprocessor_sparse.joblib` (suitable for Logistic Regression and tree-based models)
- `artifacts/preprocessing/preprocessor_dense.joblib` (recommended for KNN, dense matrix operations)

Key outputs:

- `artifacts/preprocessing/preprocessor_sparse.joblib`
- `artifacts/preprocessing/preprocessor_dense.joblib`
- `artifacts/preprocessing/feature_names.csv`
- `artifacts/data/train_test_split.json` (split policy for reproducibility)

### 03_model_logreg.ipynb — Logistic Regression Model

Purpose:

- Train and tune Logistic Regression using the saved preprocessing pipeline
- Save final model and evaluation metrics
- Generate evaluation plots (confusion matrix, ROC/PR curves where applicable)

Outputs:

- `artifacts/models/logreg_pipeline.joblib`
- `artifacts/metrics/logreg_metrics.json`
- `artifacts/plots/logreg_*.png`

### 04_model_knn.ipynb — KNN Model

Purpose:

- Train and tune KNN using the dense preprocessor
- Save final model and evaluation metrics
- Generate evaluation plots

Outputs:

- `artifacts/models/knn_pipeline.joblib`
- `artifacts/metrics/knn_metrics.json`
- `artifacts/plots/knn_*.png`

### 05_model_decision_tree.ipynb — Decision Tree Model

Purpose:

- Train and tune a Decision Tree model
- Save final model and evaluation metrics
- Generate evaluation plots + feature importance (if used)

Outputs:

- `artifacts/models/decision_tree_pipeline.joblib`
- `artifacts/metrics/decision_tree_metrics.json`
- `artifacts/plots/decision_tree_*.png`

### 06_model_random_forest.ipynb — Random Forest Model

Purpose:

- Train and tune Random Forest
- Save final model and evaluation metrics
- Generate evaluation plots + feature importance (if used)

Outputs:

- `artifacts/models/random_forest_pipeline.joblib`
- `artifacts/metrics/random_forest_metrics.json`
- `artifacts/plots/random_forest_*.png`

### 07_model_comparison.ipynb — Final Comparison and Conclusions

Purpose:

- Load metrics from all trained models
- Build a comparison table (accuracy, precision, recall, F1, ROC-AUC, etc.)
- Provide final conclusions and a justified model choice

Outputs:

- `artifacts/metrics/model_comparison.csv`
- `artifacts/reports/final_summary.md`
- `artifacts/plots/comparison_*.png` (optional)

## Notes on Reproducibility and Fair Comparison

- Preprocessing is fitted on training data only (leakage-safe)
- Splitting uses stratification to preserve class balance
- `RANDOM_STATE` is fixed in `src/config.py`
- All generated outputs are saved into `artifacts/` and overwritten on each run for consistent comparisons

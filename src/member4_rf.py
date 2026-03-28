from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.config import (
    DEFAULT_DATA_PATH,
    TARGET_COL,
    LEAKAGE_COLS,
    FORCE_CATEGORICAL_COLS,
    RANDOM_STATE,
    TEST_SIZE,
    PARAM_GRIDS,
)
from src.data_loader import load_hotel_bookings, basic_train_ready_checks
from src.preprocessing import build_preprocessor, PreprocessOptions, get_feature_names
from src.train_eval import (
    TrainOptions,
    split_xy,
    make_train_test_split,
    get_estimator,
    build_model_pipeline,
    tune_with_gridsearch,
    predict_with_optional_proba,
)
from src.metrics import compute_classification_metrics
from src.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance,
)
from src.io_utils import (
    ensure_artifact_dirs,
    save_dataframe,
    save_json,
    save_model,
    save_text,
    save_run_metadata,
)


def _to_native_float_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            if np.isnan(v):
                out[k] = None
            else:
                out[k] = float(v)
        else:
            out[k] = v
    return out


def run_member4_random_forest(
    data_path: str = DEFAULT_DATA_PATH,
    scoring: str = "f1",
    max_tune_rows: int = 40000,
) -> dict:
    dirs = ensure_artifact_dirs("artifacts")
    started_at = datetime.utcnow().isoformat() + "Z"

    df = load_hotel_bookings(data_path, drop_duplicates=True, verbose=True)
    basic_train_ready_checks(df, target_col=TARGET_COL)

    X, y = split_xy(df, TARGET_COL)
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        TrainOptions(random_state=RANDOM_STATE, test_size=TEST_SIZE),
    )

    rf_preprocess = build_preprocessor(
        drop_cols=LEAKAGE_COLS,
        force_categorical_cols=FORCE_CATEGORICAL_COLS,
        options=PreprocessOptions(
            output_sparse=True,
            scale_numeric=False,
            onehot_min_frequency=0.01,
            lower_clip_q=0.01,
            upper_clip_q=0.99,
        ),
    )

    rf_estimator = get_estimator("random_forest", random_state=RANDOM_STATE)
    rf_pipeline = build_model_pipeline(rf_preprocess, rf_estimator)

    train_options = TrainOptions(
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        cv_splits=5,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
    )
    param_grid = PARAM_GRIDS["random_forest"]
    X_tune = X_train
    y_tune = y_train
    if max_tune_rows and len(X_train) > max_tune_rows:
        sampled_idx = (
            y_train.groupby(y_train, group_keys=False)
            .apply(
                lambda s: s.sample(
                    n=max(1, int(max_tune_rows * (len(s) / len(y_train)))),
                    random_state=RANDOM_STATE,
                )
            )
            .index
        )
        X_tune = X_train.loc[sampled_idx]
        y_tune = y_train.loc[sampled_idx]

    search = tune_with_gridsearch(
        pipeline=rf_pipeline,
        param_grid=param_grid,
        X_train=X_tune,
        y_train=y_tune,
        options=train_options,
    )

    best_model = search.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred, y_proba = predict_with_optional_proba(best_model, X_test)
    test_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    test_metrics = _to_native_float_dict(test_metrics)

    save_model(best_model, dirs["models"] / "rf_pipeline.joblib")
    save_dataframe(pd.DataFrame(search.cv_results_), dirs["metrics"] / "rf_cv_results.csv")
    save_json(search.best_params_, dirs["metrics"] / "rf_best_params.json")
    save_json(test_metrics, dirs["metrics"] / "rf_test_metrics.json")
    save_dataframe(
        pd.DataFrame([test_metrics]),
        dirs["metrics"] / "rf_threshold_metrics.csv",
        index=False,
    )
    save_dataframe(
        pd.DataFrame(test_metrics["confusion_matrix"]),
        dirs["metrics"] / "rf_confusion_matrix.csv",
        index=False,
    )

    text_report = classification_report(y_test, y_pred, digits=4)
    save_text(text_report, dirs["reports"] / "rf_classification_report.txt")
    notes = (
        "# Random Forest notes\n\n"
        "- Objective: predict `is_canceled` on Hotel Booking Demand.\n"
        "- Model: RandomForestClassifier tuned with GridSearchCV.\n"
        "- Selection metric: F1 score (binary).\n"
        "- Use this file in the report discussion and viva.\n"
    )
    save_text(notes, dirs["reports"] / "rf_notes.md")

    plot_confusion_matrix(
        y_test,
        y_pred,
        title="Random Forest - Confusion Matrix",
        out_path=dirs["plots"] / "rf_confusion_matrix.png",
    )
    if y_proba is not None:
        plot_roc_curve(
            y_test,
            y_proba,
            title="Random Forest - ROC Curve",
            out_path=dirs["plots"] / "rf_roc_curve.png",
        )
        plot_pr_curve(
            y_test,
            y_proba,
            title="Random Forest - Precision-Recall Curve",
            out_path=dirs["plots"] / "rf_pr_curve.png",
        )

    model = best_model.named_steps["model"]
    preprocess = best_model.named_steps["preprocess"]
    feature_names = get_feature_names(preprocess, input_features=X_train.columns)
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": feature_names if feature_names else [f"f{i}" for i in range(len(model.feature_importances_))],
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        save_dataframe(fi, dirs["metrics"] / "rf_feature_importance.csv", index=False)
        plot_feature_importance(
            model=model,
            feature_names=fi["feature"].tolist(),
            top_n=20,
            title="Random Forest - Top 20 Feature Importance",
            out_path=dirs["plots"] / "rf_feature_importance.png",
        )

    finished_at = datetime.utcnow().isoformat() + "Z"
    run_meta = {
        "member": "member_4",
        "stage": "06_model_random_forest",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "data_path": data_path,
        "scoring": scoring,
        "max_tune_rows": max_tune_rows,
        "tune_rows_used": int(len(X_tune)),
        "best_score_cv": float(search.best_score_),
        "best_params": search.best_params_,
        "test_metrics": test_metrics,
        "artifacts_written": [
            "artifacts/models/rf_pipeline.joblib",
            "artifacts/metrics/rf_cv_results.csv",
            "artifacts/metrics/rf_best_params.json",
            "artifacts/metrics/rf_test_metrics.json",
            "artifacts/metrics/rf_threshold_metrics.csv",
            "artifacts/metrics/rf_confusion_matrix.csv",
            "artifacts/metrics/rf_feature_importance.csv",
            "artifacts/plots/rf_confusion_matrix.png",
            "artifacts/plots/rf_roc_curve.png",
            "artifacts/plots/rf_pr_curve.png",
            "artifacts/plots/rf_feature_importance.png",
            "artifacts/reports/rf_classification_report.txt",
            "artifacts/reports/rf_notes.md",
            "artifacts/reports/run_metadata.json",
        ],
    }
    save_run_metadata(run_meta, filename="run_metadata.json")

    return {
        "best_params": search.best_params_,
        "best_score_cv": float(search.best_score_),
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    results = run_member4_random_forest()
    print(json.dumps(results, indent=2))

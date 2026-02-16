import sys

sys.path.append(".")

from src.data_loader import load_hotel_bookings, basic_train_ready_checks
from src.preprocessing import build_preprocessor, PreprocessOptions
from src.train_eval import (
    split_xy,
    make_train_test_split,
    get_estimator,
    build_model_pipeline,
    evaluate_on_test,
)
from src.config import TARGET_COL, LEAKAGE_COLS

df = load_hotel_bookings("data/raw/hotel_bookings.csv")
basic_train_ready_checks(df, TARGET_COL)

X, y = split_xy(df, TARGET_COL)
X_train, X_test, y_train, y_test = make_train_test_split(X, y)

pre = build_preprocessor(
    drop_cols=LEAKAGE_COLS,
    options=PreprocessOptions(output_sparse=True, scale_numeric=True),
)

pipe = build_model_pipeline(pre, get_estimator("logreg"))
pipe.fit(X_train, y_train)

print(evaluate_on_test(pipe, X_test, y_test))

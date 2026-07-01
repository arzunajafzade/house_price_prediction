from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("housing.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "house_price_model.joblib"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
TARGET = "median_house_value"


def _stratified_split(df: pd.DataFrame):
    """Split train/test using income bands so both sets are representative."""
    income_bins = pd.cut(
        df["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, income_bins))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )
    return model


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train(tune: bool = False):
    """Train the model and save it to MODEL_PATH. Returns (model, metrics)."""
    df = pd.read_csv(DATA_PATH)

    train_df, test_df = _stratified_split(df)
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    model = _build_pipeline()

    if tune:
        param_grid = {
            "regressor__n_estimators": [100, 200],
            "regressor__max_depth": [None, 10, 20],
        }
        search = GridSearchCV(
            model, param_grid, cv=3, scoring="neg_root_mean_squared_error"
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
    else:
        model.fit(X_train, y_train)

    metrics = {
        "train": _evaluate(model, X_train, y_train),
        "test": _evaluate(model, X_test, y_test),
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model, metrics


if __name__ == "__main__":
    trained_model, results = train(tune=False)
    print("Train metrics:", results["train"])
    print("Test metrics:", results["test"])

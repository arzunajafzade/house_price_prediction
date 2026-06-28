from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


DATA_PATH = Path("housing.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "house_price_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
TARGET_COLUMN = "median_house_value"
CATEGORICAL_COLUMNS = ["ocean_proximity"]
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def stratified_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared = data.copy()
    prepared["income_cat"] = pd.cut(
        prepared["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(prepared, prepared["income_cat"]))
    train_set = prepared.loc[train_idx].drop("income_cat", axis=1)
    test_set = prepared.loc[test_idx].drop("income_cat", axis=1)
    return train_set, test_set


def build_model(feature_columns: list[str]) -> GridSearchCV:
    numeric_columns = [
        column for column in feature_columns if column not in CATEGORICAL_COLUMNS
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("regressor", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = [
        {
            "regressor__n_estimators": [30, 60],
            "regressor__max_features": [4, 6, 8],
        },
        {
            "regressor__bootstrap": [False],
            "regressor__n_estimators": [30],
            "regressor__max_features": [4, 6],
        },
    ]

    return GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )


def evaluate(model: Pipeline, features: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    mse = mean_squared_error(labels, predictions)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(labels, predictions)),
        "r2": float(r2_score(labels, predictions)),
    }


def train() -> tuple[Pipeline, dict[str, object]]:
    data = load_data()
    train_set, test_set = stratified_split(data)

    train_features = train_set.drop(TARGET_COLUMN, axis=1)
    train_labels = train_set[TARGET_COLUMN].copy()
    test_features = test_set.drop(TARGET_COLUMN, axis=1)
    test_labels = test_set[TARGET_COLUMN].copy()

    search = build_model(list(train_features.columns))
    search.fit(train_features, train_labels)
    best_model = search.best_estimator_

    metrics = {
        "best_params": search.best_params_,
        "train": evaluate(best_model, train_features, train_labels),
        "test": evaluate(best_model, test_features, test_labels),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return best_model, metrics


if __name__ == "__main__":
    _, run_metrics = train()
    print(json.dumps(run_metrics, indent=2))

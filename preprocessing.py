from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import DATA_FILE_CANDIDATES

TARGET_COLUMN = "final_sentiment"
DROP_COLUMNS = [TARGET_COLUMN, "sentiment_x_news", "net_sentiment_ratio"]
MODEL_FEATURE_COLUMNS = [
    "stock",
    "news_count",
    "sentiment_std",
    "sentiment_range",
    "sentiment_lag1",
    "sentiment_lag3",
    "sentiment_roll3",
    "sentiment_roll7",
    "sentiment_momentum",
    "month",
    "day_of_week",
]


def resolve_dataset_path():
    for path in DATA_FILE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No dataset file was found in the workspace root or data directory.")


def load_raw_dataset():
    path = resolve_dataset_path()
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.columns:
        if column in ["date", "stock"]:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["stock"] = cleaned["stock"].fillna("UNKNOWN").astype(str)

    numeric_cols = [
        column
        for column in cleaned.select_dtypes(include=[np.number]).columns
        if column not in DROP_COLUMNS
    ]

    for column in numeric_cols:
        cleaned[column] = np.where(np.isinf(cleaned[column]), np.nan, cleaned[column])
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    for column in numeric_cols:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        cleaned[column] = cleaned[column].clip(lower_bound, upper_bound)

    cleaned["month"] = cleaned["date"].dt.month.fillna(1)
    cleaned["day_of_week"] = cleaned["date"].dt.dayofweek.fillna(0)
    cleaned["year"] = cleaned["date"].dt.year.fillna(2000)

    return cleaned


def build_preprocessor(X: pd.DataFrame):
    categorical_columns = ["stock"]
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )
    return preprocessor


def prepare_model_data(df: pd.DataFrame):
    cleaned = clean_dataset(df)
    target = (cleaned[TARGET_COLUMN] >= 0).astype(int)
    feature_frame = cleaned[MODEL_FEATURE_COLUMNS].copy()
    return feature_frame, target


def split_training_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def summary_of_missing_values(df: pd.DataFrame):
    return df.isna().sum().sort_values(ascending=False)

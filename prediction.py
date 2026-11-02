from __future__ import annotations

import math

import joblib
import pandas as pd

from config import MODEL_PATH
from model_training import train_models
from preprocessing import clean_dataset, load_raw_dataset


def load_model():
    if not MODEL_PATH.exists():
        train_models()
    return joblib.load(MODEL_PATH)


def default_prediction_values():
    raw_df = load_raw_dataset()
    prepared = clean_dataset(raw_df)
    defaults = {
        "stock": prepared["stock"].mode().iloc[0],
        "date": prepared["date"].max().strftime("%Y-%m-%d"),
        "news_count": float(prepared["news_count"].median()),
        "sentiment_std": float(prepared["sentiment_std"].median()),
        "sentiment_range": float(prepared["sentiment_range"].median()),
        "sentiment_lag1": float(prepared["sentiment_lag1"].median()),
        "sentiment_lag3": float(prepared["sentiment_lag3"].median()),
        "sentiment_roll3": float(prepared["sentiment_roll3"].median()),
        "sentiment_roll7": float(prepared["sentiment_roll7"].median()),
        "sentiment_momentum": float(prepared["sentiment_momentum"].median()),
    }
    return defaults


def make_prediction_from_dict(values):
    model = load_model()
    record = {
        "stock": values["stock"],
        "news_count": float(values["news_count"]),
        "sentiment_std": float(values.get("sentiment_std", 0.0)),
        "sentiment_range": float(values.get("sentiment_range", 0.0)),
        "sentiment_lag1": float(values.get("sentiment_lag1", 0.0)),
        "sentiment_lag3": float(values.get("sentiment_lag3", 0.0)),
        "sentiment_roll3": float(values.get("sentiment_roll3", 0.0)),
        "sentiment_roll7": float(values.get("sentiment_roll7", 0.0)),
        "sentiment_momentum": float(values.get("sentiment_momentum", 0.0)),
        "month": pd.to_datetime(values["date"]).month,
        "day_of_week": pd.to_datetime(values["date"]).dayofweek,
    }

    data = pd.DataFrame([record])
    predicted_label = int(model.predict(data)[0])
    probability = model.predict_proba(data)[0]
    positive_probability = float(probability[1]) if len(probability) > 1 else float(probability[0])
    sentiment_text = "Positive sentiment" if predicted_label == 1 else "Negative sentiment"
    return {
        "label": predicted_label,
        "sentiment_text": sentiment_text,
        "probability": positive_probability,
        "confidence": max(positive_probability, 1 - positive_probability),
    }

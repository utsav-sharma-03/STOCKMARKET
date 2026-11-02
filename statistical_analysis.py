from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe_variable(series: pd.Series):
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "mode": float(series.mode().iloc[0]) if not series.mode().empty else None,
        "variance": float(series.var(ddof=1)) if series.count() > 1 else 0.0,
        "std_dev": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def confidence_interval(values: pd.Series, confidence_level: float = 0.95):
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return {"mean": float(np.mean(values)) if len(values) else None, "lower": None, "upper": None}

    mean_value = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * standard_error
    return {
        "mean": mean_value,
        "lower": mean_value - margin,
        "upper": mean_value + margin,
    }


def hypothesis_test_summary(df: pd.DataFrame):
    sentiment_label = (df["final_sentiment"] >= 0).astype(int)
    positive = df.loc[sentiment_label == 1, "final_sentiment"]
    negative = df.loc[sentiment_label == 0, "final_sentiment"]

    if len(positive) > 1 and len(negative) > 1:
        t_stat, p_value = stats.ttest_ind(positive, negative, equal_var=False)
    else:
        t_stat, p_value = None, None

    if "stock" in df.columns:
        contingency = pd.crosstab(df["stock"], sentiment_label)
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency) if contingency.shape[0] > 1 and contingency.shape[1] > 1 else (None, None, None, None)
    else:
        chi2_stat, chi2_p = None, None

    return {
        "positive_mean": float(positive.mean()) if not positive.empty else None,
        "negative_mean": float(negative.mean()) if not negative.empty else None,
        "t_stat": float(t_stat) if t_stat is not None else None,
        "t_p_value": float(p_value) if p_value is not None else None,
        "chi2_stat": float(chi2_stat) if chi2_stat is not None else None,
        "chi2_p_value": float(chi2_p) if chi2_p is not None else None,
    }


def probability_summary(df: pd.DataFrame):
    sentiment_label = (df["final_sentiment"] >= 0).astype(int)
    positive_probability = float(sentiment_label.mean())
    negative_probability = 1 - positive_probability
    return {
        "positive_probability": positive_probability,
        "negative_probability": negative_probability,
        "conditional_positive_given_stock": df.groupby("stock")["final_sentiment"].apply(lambda s: (s >= 0).mean()).to_dict() if "stock" in df.columns else {},
    }

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns


def dataset_overview(df: pd.DataFrame):
    overview = {
        "records": int(df.shape[0]),
        "features": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "stocks": int(df["stock"].nunique()) if "stock" in df.columns else 0,
        "date_start": df["date"].min() if "date" in df.columns else None,
        "date_end": df["date"].max() if "date" in df.columns else None,
    }
    return overview


def describe_columns(df: pd.DataFrame, columns):
    return df[columns].describe().T


def correlation_heatmap(df: pd.DataFrame, columns=None):
    plot_data = df if columns is None else df[columns]
    numeric = plot_data.select_dtypes(include=["number"])
    corr = numeric.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[[0, "#f5f5f5"], [0.5, "#8a8a8a"], [1, "#111111"]],
        title="Correlation Heatmap",
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#0c0c0c", plot_bgcolor="#0c0c0c", font_color="#ffffff")
    return fig


def histogram_plot(df: pd.DataFrame, column: str):
    fig = px.histogram(
        df,
        x=column,
        nbins=30,
        marginal="box",
        title=f"Distribution of {column}",
        color_discrete_sequence=["#d4d4d4"],
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#0c0c0c", plot_bgcolor="#0c0c0c", font_color="#ffffff")
    return fig


def boxplot_plot(df: pd.DataFrame, column: str):
    fig = px.box(
        df,
        x=column,
        title=f"Boxplot for {column}",
        color_discrete_sequence=["#d4d4d4"],
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#0c0c0c", plot_bgcolor="#0c0c0c", font_color="#ffffff")
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="stock" if "stock" in df.columns else None,
        title=f"{y_col} vs {x_col}",
        opacity=0.8,
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="#0c0c0c", plot_bgcolor="#0c0c0c", font_color="#ffffff")
    return fig


def stock_sentiment_trend(df: pd.DataFrame):
    if "stock" not in df.columns or "final_sentiment" not in df.columns:
        return None
    summary = (
        df.groupby("stock", as_index=False)["final_sentiment"]
        .mean()
        .sort_values("final_sentiment", ascending=False)
    )
    fig = px.bar(summary, x="stock", y="final_sentiment", title="Average Sentiment by Stock")
    fig.update_layout(template="plotly_white", paper_bgcolor="#0c0c0c", plot_bgcolor="#0c0c0c", font_color="#ffffff")
    return fig


def plot_seaborn_hist(df: pd.DataFrame, column: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column].dropna(), bins=30, color="#d4d4d4", edgecolor="#ffffff", ax=ax)
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.tick_params(colors="#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    return fig

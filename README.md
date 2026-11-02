# News Sentiment Forecasting Project

This project analyzes the uploaded stock-news sentiment dataset and builds a machine learning workflow to predict whether a stock-day sentiment signal is positive or negative.

## Project objective

The dataset contains stock and sentiment indicators derived from news coverage. The goal is to understand the patterns in those indicators and build a classification model that predicts the direction of sentiment using features such as news volume, sentiment momentum, rolling averages, lag variables, and stock identity.

## Dataset overview

The dataset includes:

- `date`
- `stock`
- `final_sentiment`
- `news_count`
- `sentiment_std`
- `sentiment_x_news`
- `sentiment_range`
- `sentiment_lag1`
- `sentiment_lag3`
- `sentiment_roll3`
- `sentiment_roll7`
- `sentiment_momentum`
- `net_sentiment_ratio`

The target used for modeling is the sign of `final_sentiment` (positive vs negative). This is a classification problem aligned with the real-world goal of identifying likely positive or negative sentiment conditions in stock-related news.

## Methodology

1. Load and clean the data.
2. Remove duplicate rows and impute missing values.
3. Convert dates into useful temporal features.
4. Clip extreme outliers using the interquartile range.
5. Encode the stock identifier and scale numeric features.
6. Train and compare multiple classifiers.
7. Save the best model to the `models` directory.
8. Expose predictions through a Streamlit dashboard.

## Installation

```bash
pip install -r requirements.txt
```

## Launch the app

```bash
python -m streamlit run app.py
```

## Files

- `app.py` — Streamlit frontend
- `config.py` — project paths and metadata
- `preprocessing.py` — cleaning, feature creation, and pipeline setup
- `eda.py` — EDA visualizations
- `statistical_analysis.py` — descriptive and inferential statistics
- `model_training.py` — training, model comparison, evaluation, and persistence
- `prediction.py` — input generation and prediction logic
- `news_features.csv` — uploaded dataset
- `models/best_model.pkl` — trained model

## Notes

The app is intentionally designed with a strict black-and-white premium visual style and uses a practical sentiment-classification workflow rather than a mock dashboard.

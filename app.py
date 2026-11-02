from __future__ import annotations

import pandas as pd
import streamlit as st

from eda import (
    boxplot_plot,
    correlation_heatmap,
    dataset_overview,
    histogram_plot,
    scatter_plot,
    stock_sentiment_trend,
)
from model_training import train_models
from prediction import default_prediction_values, make_prediction_from_dict
from preprocessing import clean_dataset, load_raw_dataset
from statistical_analysis import confidence_interval, describe_variable, hypothesis_test_summary, probability_summary

st.set_page_config(page_title="News Sentiment Intelligence", page_icon="◌", layout="wide")

NAV_ITEMS = [
    "Dashboard",
    "Dataset Overview",
    "Data Preprocessing",
    "Exploratory Data Analysis",
    "Statistical Analysis",
    "Machine Learning Models",
    "Prediction Center",
    "About Project",
]

st.markdown(
    """
    <style>
    :root {
        --bg: #f3efe7;
        --bg-soft: #efe9df;
        --card: #faf7f1;
        --card-strong: #f6f1e9;
        --panel: #f9f5ef;
        --border: #d7d0c1;
        --text: #1c1a17;
        --muted: #5d594f;
        --olive: #738b62;
        --olive-dark: #4d5d3f;
        --olive-soft: #e3ebdf;
        --shadow: rgba(35, 30, 20, 0.08);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
    }

    .stApp {
        background: linear-gradient(180deg, #f5f2eb 0%, #f2efe8 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0ebdf 0%, #ece4d7 100%);
        border-right: 1px solid var(--border);
        box-shadow: inset -1px 0 0 rgba(0, 0, 0, 0.02);
    }

    [data-testid="stSidebarNav"] {
        padding-top: 0.8rem;
    }

    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding: 0.8rem 0.8rem 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.9rem;
    }

    .sidebar-logo {
        width: 30px;
        height: 30px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, var(--olive) 0%, var(--olive-dark) 100%);
        color: #ffffff;
        font-size: 0.8rem;
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(115, 139, 98, 0.25);
    }

    .sidebar-name {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text);
    }

    .section-header {
        margin: 0.2rem 0 1.5rem;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid var(--border);
    }

    .section-header h2 {
        margin: 0;
        font-size: clamp(2rem, 3vw, 3rem);
        letter-spacing: -0.06em;
        font-weight: 700;
        color: var(--text);
    }

    .section-header p {
        margin: 0.5rem 0 0;
        color: var(--muted);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .metric-card {
        background: linear-gradient(180deg, #fffdf9 0%, #f7f2ea 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.1rem 1.15rem;
        height: 100%;
        min-height: 128px;
        box-shadow: 0 10px 22px var(--shadow);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: clamp(1.7rem, 2vw, 2.3rem);
        line-height: 1.1;
        font-weight: 700;
        color: var(--text);
    }

    .metric-delta {
        margin-top: 0.6rem;
        font-size: 0.78rem;
        color: var(--muted);
    }

    .content-card {
        background: linear-gradient(180deg, #fffdf9 0%, #f6f1e8 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        height: 100%;
        box-shadow: 0 10px 20px var(--shadow);
    }

    .panel-title {
        margin: 0 0 0.8rem;
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .subtle-text {
        color: var(--muted);
        line-height: 1.7;
        font-size: 0.96rem;
    }

    .stButton > button {
        background: linear-gradient(180deg, var(--olive) 0%, var(--olive-dark) 100%);
        color: #ffffff;
        border: 1px solid rgba(77, 93, 63, 0.4);
        border-radius: 12px;
        font-weight: 600;
        padding: 0.65rem 1.25rem;
        box-shadow: 0 10px 18px rgba(115, 139, 98, 0.25);
    }

    .stButton > button:hover {
        background: linear-gradient(180deg, #7d926f 0%, #536a4d 100%);
        border-color: rgba(77, 93, 63, 0.6);
    }

    div[data-testid="stSidebarNavLink"] {
        color: var(--text) !important;
        border-radius: 10px;
    }

    div[data-testid="stSidebarNavLink"] span,
    div[data-testid="stSidebarNavLink"] p,
    div[data-testid="stSidebarNavLink"] div {
        color: var(--text) !important;
    }

    div[data-testid="stSidebarNavLink"][aria-current="page"] {
        background: rgba(115, 139, 98, 0.12);
        border: 1px solid rgba(115, 139, 98, 0.25);
    }

    .stSelectbox > div > div,
    .stDateInput > div > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background: #fffdfb;
        border: 1px solid var(--border);
        border-radius: 10px;
        color: var(--text);
    }

    .stSelectbox label,
    .stDateInput label,
    .stNumberInput label,
    .stTextInput label,
    .stRadio label,
    .stCheckbox label,
    div[data-testid="stForm"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stDateInput"] label {
        color: var(--text) !important;
        font-weight: 600;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
    }

    div[data-testid="stForm"] *,
    div[data-testid="stNumberInput"] *,
    div[data-testid="stSelectbox"] *,
    div[data-testid="stDateInput"] * {
        color: var(--text) !important;
    }

    div[data-testid="stForm"] input,
    div[data-testid="stForm"] .stNumberInput input,
    div[data-testid="stForm"] .stSelectbox > div > div,
    div[data-testid="stForm"] .stDateInput > div > div {
        color: var(--text) !important;
        background-color: #fffdfb !important;
    }

    .stDataFrame, .stTable {
        background: transparent;
        color: var(--text);
    }

    .stDataFrame > div > div > table,
    .stTable > div > div > table {
        border-collapse: collapse;
        color: var(--text);
    }

    .stDataFrame [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown p, .stMarkdown li,
    .stMarkdown strong {
        color: var(--text);
    }

    .stTabs [role="tablist"] {
        gap: 0.6rem;
    }

    .stTabs [role="tab"] {
        background: transparent;
        color: var(--muted);
        border: 1px solid var(--border);
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 0.85rem;
    }

    .stTabs [role="tab"][aria-selected="true"] {
        color: var(--text);
        background: rgba(255, 255, 255, 0.35);
        border-bottom-color: transparent;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: var(--text);
    }

    .stAlert {
        background: #eef3ea;
        border: 1px solid rgba(115, 139, 98, 0.25);
        color: var(--text);
    }

    .block-container > div {
        gap: 1rem;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-top: 1.2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_clean_dataset():
    return clean_dataset(load_raw_dataset())


@st.cache_data
def get_model_output():
    return train_models()


def render_page_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str | int, delta: str | None = None):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_page(df: pd.DataFrame):
    overview = dataset_overview(df)
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    render_page_header("Dashboard", "AI-driven stock sentiment monitoring and forecast intelligence")

    st.markdown(
        """
        <div class="subtle-text" style="margin-bottom: 1.2rem; max-width: 820px;">
            This project analyzes news-driven sentiment features to classify whether a company-day signal is likely to be positive or negative.
            The dashboard combines data quality checks, performance summaries, and machine learning outputs in a single decision-support interface.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Total records", overview["records"], "Dataset size")
    with metric_cols[1]:
        render_metric_card("Features", overview["features"], "Engineered indicators")
    with metric_cols[2]:
        render_metric_card("Missing values", overview["missing_values"], "Imputed or cleaned")
    with metric_cols[3]:
        render_metric_card("Unique stocks", overview["stocks"], "Coverage across firms")

    info_col_left, info_col_right = st.columns([1.4, 1])
    with info_col_left:
        st.markdown(
            f"""
            <div class="content-card">
                <div class="panel-title">Dataset summary</div>
                <div class="subtle-text">
                    <strong>Start date:</strong> {overview['date_start']} <br>
                    <strong>End date:</strong> {overview['date_end']} <br>
                    <strong>Observation focus:</strong> stock-specific sentiment signals driven by news activity and rolling sentiment windows.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col_right:
        st.markdown(
            """
            <div class="content-card">
                <div class="panel-title">Analytical focus</div>
                <div class="subtle-text">
                    The model predicts the direction of a sentiment signal using a compact set of features including news count, lagged sentiment,
                    momentum, and rolling averages. This makes the application practical for early sentiment regime detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("<div class='panel-title'>Feature relationship</div>", unsafe_allow_html=True)
        if numeric_columns:
            st.plotly_chart(correlation_heatmap(df, numeric_columns[:8]), use_container_width=True)
    with chart_right:
        st.markdown("<div class='panel-title'>Stock sentiment profile</div>", unsafe_allow_html=True)
        if "stock" in df.columns:
            st.plotly_chart(stock_sentiment_trend(df), use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>Recent records</div>", unsafe_allow_html=True)
    st.dataframe(df.head(12), use_container_width=True)


def render_dataset_overview_page(df: pd.DataFrame):
    render_page_header("Dataset Overview", "Structured view of the available stock-news features")

    stock_choice = st.selectbox("Filter by stock", ["All"] + sorted(df["stock"].unique().tolist()))
    filtered = df if stock_choice == "All" else df[df["stock"] == stock_choice]
    st.dataframe(filtered.head(50), use_container_width=True)

    selected_columns = st.multiselect(
        "Choose columns to inspect",
        df.columns.tolist(),
        default=df.columns[:6].tolist(),
    )
    if selected_columns:
        st.dataframe(df[selected_columns].describe().T, use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        selected_numeric = st.selectbox("Explore numeric feature", numeric_cols)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(histogram_plot(df, selected_numeric), use_container_width=True)
        with col_b:
            st.plotly_chart(boxplot_plot(df, selected_numeric), use_container_width=True)

    st.markdown("<div class='panel-title'>Data types</div>", unsafe_allow_html=True)
    st.dataframe(df.dtypes.rename("dtype").reset_index().rename(columns={"index": "column", "dtype": "type"}), use_container_width=True)


def render_home_page(df: pd.DataFrame):
    render_dashboard_page(df)


def render_exploration_page(df: pd.DataFrame):
    render_dataset_overview_page(df)


def render_preprocessing_page(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame):
    render_page_header("Data Preprocessing", "Quality controls and transformations applied before modeling")

    missing_before = raw_df.isna().sum().sort_values(ascending=False)
    missing_after = cleaned_df.isna().sum().sort_values(ascending=False)

    before, after = st.columns(2)
    with before:
        st.markdown("<div class='panel-title'>Before preprocessing</div>", unsafe_allow_html=True)
        st.dataframe(missing_before.to_frame(name="missing_values"), use_container_width=True)
    with after:
        st.markdown("<div class='panel-title'>After preprocessing</div>", unsafe_allow_html=True)
        st.dataframe(missing_after.to_frame(name="missing_values"), use_container_width=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    quality_cols = st.columns(3)
    with quality_cols[0]:
        render_metric_card("Rows before", raw_df.shape[0])
    with quality_cols[1]:
        render_metric_card("Rows after", cleaned_df.shape[0])
    with quality_cols[2]:
        render_metric_card("Duplicates removed", int(raw_df.duplicated().sum()))

    st.markdown(
        """
        <div class="content-card" style="margin-top: 1rem;">
            <div class="panel-title">Transformation summary</div>
            <div class="subtle-text">
                • Missing values were addressed with data-aware imputation strategies for numeric indicators.<br>
                • Duplicate rows were removed to preserve the integrity of the learning dataset.<br>
                • Outlier pressure was reduced using clipping rules to avoid model distortion.<br>
                • Date-based features were preserved so the model can learn temporal patterns such as month and weekday effects.<br>
                • Categorical stock identity was retained and encoded during the modeling pipeline.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_eda_page(df: pd.DataFrame):
    render_page_header("Exploratory Data Analysis", "Statistical inspection of shapes, distributions, and relationships")
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        selected_numeric = st.selectbox("Numeric field", numeric_columns)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(histogram_plot(df, selected_numeric), use_container_width=True)
        with col_b:
            st.plotly_chart(boxplot_plot(df, selected_numeric), use_container_width=True)

        scatter_x = st.selectbox("Scatter X", numeric_columns)
        scatter_y = st.selectbox("Scatter Y", [column for column in numeric_columns if column != scatter_x], index=0)
        st.plotly_chart(scatter_plot(df, scatter_x, scatter_y), use_container_width=True)

        st.plotly_chart(correlation_heatmap(df, numeric_columns), use_container_width=True)

    if "stock" in df.columns:
        st.plotly_chart(stock_sentiment_trend(df), use_container_width=True)


def render_statistics_page(df: pd.DataFrame):
    render_page_header("Statistical Analysis", "Descriptive and inferential summaries for feature interpretation")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    selected_col = st.selectbox("Choose variable", numeric_cols)

    st.markdown("<div class='panel-title'>Summary statistics</div>", unsafe_allow_html=True)
    st.json(describe_variable(df[selected_col]))

    ci = confidence_interval(df[selected_col], confidence_level=0.95)
    st.markdown("<div class='panel-title' style='margin-top: 1rem;'>Confidence interval</div>", unsafe_allow_html=True)
    st.write(ci)

    st.markdown("<div class='panel-title' style='margin-top: 1rem;'>Hypothesis testing</div>", unsafe_allow_html=True)
    st.json(hypothesis_test_summary(df))

    st.markdown("<div class='panel-title' style='margin-top: 1rem;'>Probability analysis</div>", unsafe_allow_html=True)
    st.json(probability_summary(df))


def render_ml_page():
    render_page_header("Machine Learning Models", "Model comparison and decision quality metrics")
    model_summary = get_model_output()
    results = pd.DataFrame(model_summary["results"])
    st.dataframe(results, use_container_width=True)

    best_name = model_summary["best_model_name"]
    st.success(f"Best model selected: {best_name}")

    X_test = model_summary["X_test"]
    y_test = model_summary["y_test"]
    best_pipeline = model_summary["best_pipeline"]
    predictions = best_pipeline.predict(X_test)

    confusion = model_summary["confusion_matrix"]
    st.markdown("<div class='panel-title' style='margin-top: 1rem;'>Confusion matrix</div>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(confusion, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"]),
        use_container_width=True,
    )

    st.markdown("<div class='panel-title' style='margin-top: 1rem;'>Feature importance</div>", unsafe_allow_html=True)
    importance_df = pd.DataFrame(model_summary["feature_importance"]).head(10)
    st.dataframe(importance_df, use_container_width=True)

    actual_vs_pred = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    st.markdown("<div class='panel-title' style='margin-top: 1rem;'>Prediction vs. actual</div>", unsafe_allow_html=True)
    st.scatter_chart(actual_vs_pred)


def render_prediction_page(df: pd.DataFrame):
    render_page_header("Prediction Center", "Estimate the expected sentiment direction using the trained model")
    default_values = default_prediction_values()
    with st.form("prediction_form"):
        stock = st.selectbox("Stock", sorted(df["stock"].unique().tolist()), index=0)
        date_value = st.date_input("Date", pd.to_datetime(default_values["date"]))
        numeric_fields = {
            "news_count": st.number_input("News count", min_value=0.0, value=float(default_values["news_count"]), step=1.0),
            "sentiment_std": st.number_input("Sentiment std", value=float(default_values["sentiment_std"]), step=0.01),
            "sentiment_range": st.number_input("Sentiment range", value=float(default_values["sentiment_range"]), step=0.01),
            "sentiment_lag1": st.number_input("Sentiment lag 1", value=float(default_values["sentiment_lag1"]), step=0.01),
            "sentiment_lag3": st.number_input("Sentiment lag 3", value=float(default_values["sentiment_lag3"]), step=0.01),
            "sentiment_roll3": st.number_input("Sentiment roll 3", value=float(default_values["sentiment_roll3"]), step=0.01),
            "sentiment_roll7": st.number_input("Sentiment roll 7", value=float(default_values["sentiment_roll7"]), step=0.01),
            "sentiment_momentum": st.number_input("Sentiment momentum", value=float(default_values["sentiment_momentum"]), step=0.01),
        }
        submitted = st.form_submit_button("Generate prediction")

    if submitted:
        payload = {
            "stock": stock,
            "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
            **numeric_fields,
        }
        result = make_prediction_from_dict(payload)

        st.markdown(
            """
            <div class="content-card" style="margin-top: 1rem;">
                <div class="panel-title" style="color: #1c1a17;">Prediction result</div>
                <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem; color: #1c1a17;">{sentiment}</div>
                <div style="color: #5d594f; line-height: 1.7;">Confidence: {confidence}%</div>
                <div style="color: #5d594f; line-height: 1.7;">Estimated positive probability: {probability}%</div>
            </div>
            """.format(
                sentiment=result["sentiment_text"],
                confidence=f"{result['confidence'] * 100:.1f}",
                probability=f"{result['probability'] * 100:.2f}",
            ),
            unsafe_allow_html=True,
        )


def render_about_page():
    render_page_header("About Project", "Context, methodology, and the intent behind the dashboard")
    st.markdown(
        """
        <div class="content-card">
            <div class="subtle-text">
                This project is designed to answer a practical forecasting question: whether news-derived sentiment signals can help identify positive or negative sentiment conditions for a stock-day event.<br><br>
                <strong>Objective:</strong> classify sentiment regime using engineered stock-news features.<br>
                <strong>Dataset:</strong> the uploaded stock sentiment dataset provided in the workspace.<br>
                <strong>Technologies:</strong> Python, pandas, scikit-learn, seaborn, plotly, streamlit, joblib, and scipy.<br>
                <strong>Methods:</strong> data cleaning, feature engineering, exploratory analysis, statistical testing, model comparison, and prediction.<br>
                <strong>Models:</strong> Logistic Regression, Random Forest, and Gradient Boosting.<br>
                <strong>Limitations:</strong> this is sentiment-first analysis and does not yet incorporate full price, volume, or macroeconomic context.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    df = get_clean_dataset()
    raw_df = load_raw_dataset()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="sidebar-logo">NS</div>
                <div class="sidebar-name">News Sentiment</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        navigation = st.radio("", NAV_ITEMS, index=0, label_visibility="collapsed")

    if navigation == "Dashboard":
        render_dashboard_page(df)
    elif navigation == "Dataset Overview":
        render_dataset_overview_page(df)
    elif navigation == "Data Preprocessing":
        render_preprocessing_page(raw_df, df)
    elif navigation == "Exploratory Data Analysis":
        render_eda_page(df)
    elif navigation == "Statistical Analysis":
        render_statistics_page(df)
    elif navigation == "Machine Learning Models":
        render_ml_page()
    elif navigation == "Prediction Center":
        render_prediction_page(df)
    else:
        render_about_page()


if __name__ == "__main__":
    main()

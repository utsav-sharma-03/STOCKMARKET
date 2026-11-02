from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from config import MODEL_DIR, MODEL_META_PATH, MODEL_PATH
from preprocessing import build_preprocessor, prepare_model_data, split_training_data


def train_models():
    df = pd.read_csv(Path(__file__).resolve().parent / "news_features.csv")
    X, y = prepare_model_data(df)
    X_train, X_test, y_train, y_test = split_training_data(X, y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    best_pipeline = None
    best_metric = -1
    best_name = ""

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append(
            {
                "Model": model_name,
                "Accuracy": round(float(accuracy), 4),
                "Precision": round(float(precision), 4),
                "Recall": round(float(recall), 4),
                "F1 Score": round(float(f1), 4),
            }
        )

        if f1 > best_metric:
            best_metric = f1
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise ValueError("No model could be trained on the uploaded dataset.")

    model_predictions = best_pipeline.predict(X_test)
    confusion = confusion_matrix(y_test, model_predictions)
    report = classification_report(y_test, model_predictions, output_dict=True, zero_division=0)

    feature_importance = []
    if hasattr(best_pipeline.named_steps["model"], "feature_importances_"):
        preprocessor = best_pipeline.named_steps["preprocessor"]
        transformed_names = preprocessor.get_feature_names_out()
        importances = best_pipeline.named_steps["model"].feature_importances_
        for name, score in zip(transformed_names, importances):
            feature_importance.append({"feature": name, "importance": round(float(score), 4)})
    elif hasattr(best_pipeline.named_steps["model"], "coef_"):
        preprocessor = best_pipeline.named_steps["preprocessor"]
        transformed_names = preprocessor.get_feature_names_out()
        coefficients = best_pipeline.named_steps["model"].coef_.ravel()
        for name, score in zip(transformed_names, coefficients):
            feature_importance.append({"feature": name, "importance": round(float(score), 4)})

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    with open(MODEL_META_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_model": best_name,
                "features": X.columns.tolist(),
                "confusion_matrix": confusion.tolist(),
                "classification_report": report,
                "feature_importance": feature_importance,
            },
            file,
            indent=2,
        )

    return {
        "results": results,
        "best_model_name": best_name,
        "best_pipeline": best_pipeline,
        "X_test": X_test,
        "y_test": y_test,
        "feature_importance": sorted(feature_importance, key=lambda item: item["importance"], reverse=True),
        "confusion_matrix": confusion,
    }

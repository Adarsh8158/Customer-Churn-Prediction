from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"
METRICS_PATH = ROOT / "model" / "metrics.json"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        median_total_charges = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": {
            key: {inner_key: float(val) if isinstance(val, (int, float)) else val for inner_key, val in value.items()}
            for key, value in report.items()
            if isinstance(value, dict)
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_dataset(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("The dataset must contain a 'Churn' target column.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    test_metrics = evaluate_model(pipeline, X_test, y_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

    results = {
        "test_metrics": test_metrics,
        "cross_validation_roc_auc_mean": float(cv_scores.mean()),
        "cross_validation_roc_auc_std": float(cv_scores.std()),
        "cross_validation_roc_auc_scores": [float(score) for score in cv_scores],
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Model trained and saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print(json.dumps({
        "accuracy": round(results["test_metrics"]["accuracy"], 4),
        "roc_auc": round(results["test_metrics"]["roc_auc"], 4),
        "cv_roc_auc_mean": round(results["cross_validation_roc_auc_mean"], 4),
    }, indent=2))


if __name__ == "__main__":
    main()

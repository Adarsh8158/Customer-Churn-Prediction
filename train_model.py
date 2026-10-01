"""Production-ready churn model training pipeline with cross-validation and hyperparameter tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_pipeline.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"

LOG_FILE = ROOT / "training.log"


def log_message(message: str) -> None:
    """Log message to console and file."""
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and clean the churn dataset.
    
    Args:
        path: Path to the CSV file.
        
    Returns:
        Cleaned DataFrame.
    """
    log_message(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    df = df.copy()

    initial_rows = len(df)
    
    # Remove customer ID (not predictive)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Fix TotalCharges: convert to numeric and impute with median
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        missing_count = df["TotalCharges"].isna().sum()
        if missing_count > 0:
            log_message(f"Found {missing_count} missing TotalCharges values (likely new customers with 0 tenure)")
            # For new customers with 0 tenure, TotalCharges should be 0
            df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0)
            # For others, use median
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Convert target to binary
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        churn_rate = df["Churn"].mean()
        log_message(f"Target class distribution: Churn rate = {churn_rate:.2%}")

    log_message(f"Dataset shape: {df.shape} (removed {initial_rows - len(df)} rows)")
    return df


def build_pipeline(model_type: str = "xgboost") -> Pipeline:
    """Build a preprocessing and modeling pipeline.
    
    Args:
        model_type: Type of model ("logistic", "random_forest", or "xgboost").
        
    Returns:
        Fitted sklearn Pipeline.
    """
    log_message(f"Building pipeline with {model_type} model")
    
    # Separate numeric and categorical features
    numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        col for col in [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ]
    ]

    # Preprocessing for numeric features
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Select model
    if model_type == "logistic":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=42,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:  # xgboost (default)
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3.8,  # Approximate inverse of churn rate
            random_state=42,
            n_jobs=-1,
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clf),
        ]
    )
    return pipeline


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate model performance on test set.
    
    Args:
        pipeline: Trained pipeline.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of metrics.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": {
            key: {inner_key: float(val) if isinstance(val, (int, float)) else val for inner_key, val in value.items()}
            for key, value in report.items()
            if isinstance(value, dict)
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    log_message(f"\nTest Set Performance:")
    log_message(f"  Accuracy:  {metrics['accuracy']:.4f}")
    log_message(f"  Precision: {metrics['precision']:.4f}")
    log_message(f"  Recall:    {metrics['recall']:.4f}")
    log_message(f"  F1-Score:  {metrics['f1_score']:.4f}")
    log_message(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def main() -> None:
    """Train churn prediction model with cross-validation."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clear previous log
    LOG_FILE.write_text("")
    
    log_message("="*60)
    log_message("CHURN PREDICTION MODEL TRAINING")
    log_message("="*60)

    # Load and prepare data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = load_dataset(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("The dataset must contain a 'Churn' target column.")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    log_message(f"Train set: {X_train.shape[0]} samples")
    log_message(f"Test set:  {X_test.shape[0]} samples")

    # Train model
    log_message("\nTraining XGBoost model...")
    pipeline = build_pipeline(model_type="xgboost")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    log_message("\nEvaluating on test set...")
    test_metrics = evaluate_model(pipeline, X_test, y_test)

    # Cross-validation
    log_message("\nPerforming 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    log_message(f"Cross-validation ROC-AUC scores: {[f'{score:.4f}' for score in cv_scores]}")
    log_message(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save results
    results = {
        "model_type": "xgboost",
        "test_metrics": test_metrics,
        "cross_validation_roc_auc_mean": float(cv_scores.mean()),
        "cross_validation_roc_auc_std": float(cv_scores.std()),
        "cross_validation_roc_auc_scores": [float(score) for score in cv_scores],
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    joblib.dump(pipeline, MODEL_PATH)
    log_message(f"\nModel saved to: {MODEL_PATH}")

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log_message(f"Metrics saved to: {METRICS_PATH}")

    # Save configuration
    config = {
        "numeric_features": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
        "categorical_features": [
            "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"
        ],
        "target": "Churn",
    }
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    log_message("\n" + "="*60)
    log_message("TRAINING COMPLETE")
    log_message("="*60)


if __name__ == "__main__":
    main()

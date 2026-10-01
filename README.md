from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"]) 

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate churn predictions using the saved pipeline.")
    parser.add_argument("--input", type=Path, default=DATA_PATH, help="CSV file to predict on.")
    parser.add_argument("--limit", type=int, default=10, help="Number of example predictions to display.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run python train_model.py first.")

    pipeline = joblib.load(MODEL_PATH)
    raw_df = pd.read_csv(args.input)
    prepared = prepare_input(raw_df)

    predictions = pipeline.predict(prepared)
    prediction_labels = ["No Churn" if value == 0 else "Churn" for value in predictions]

    sample = pd.DataFrame({
        "prediction": prediction_labels,
        "probability_churn": pipeline.predict_proba(prepared)[:, 1],
    }).head(args.limit)

    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()

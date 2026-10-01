"""Generate predictions using the trained churn model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw input data for prediction.
    
    Args:
        df: Raw input DataFrame.
        
    Returns:
        Cleaned DataFrame ready for prediction.
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.loc[df["tenure"] == 0, "TotalCharges"] = 0
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Remove target if present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    return df


def predict(input_path: Path | str, output_path: Path | str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Generate predictions on input data.
    
    Args:
        input_path: Path to CSV file.
        output_path: Optional path to save predictions.
        limit: Optional limit on number of rows to display.
        
    Returns:
        DataFrame with predictions.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first."
        )

    print(f"Loading model from {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    print(f"Loading data from {input_path}")
    raw_df = pd.read_csv(input_path)
    prepared = prepare_input(raw_df)

    print(f"Generating predictions for {len(prepared)} customers...")
    predictions = pipeline.predict(prepared)
    probabilities = pipeline.predict_proba(prepared)[:, 1]

    results = pd.DataFrame({
        "prediction": ["No Churn" if p == 0 else "Churn" for p in predictions],
        "churn_probability": probabilities,
        "churn_risk": pd.cut(probabilities, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"]),
    })

    # Combine with original data
    if "customerID" in raw_df.columns:
        results.insert(0, "customerID", raw_df["customerID"].values)

    if output_path:
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    display_limit = limit or 10
    print(f"\nTop {min(display_limit, len(results))} predictions:")
    print(results.head(display_limit).to_string(index=False))

    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Total customers: {len(results)}")
    print(f"  Predicted churners: {(results['prediction'] == 'Churn').sum()} ({(results['prediction'] == 'Churn').mean():.2%})")
    print(f"  Average churn probability: {probabilities.mean():.4f}")
    print(f"\nChurn Risk Distribution:")
    print(results['churn_risk'].value_counts().sort_index())

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate churn predictions using the trained model."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_PATH),
        help="Path to input CSV file (default: data/WA_Fn-UseC_-Telco-Customer-Churn.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions CSV (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of predictions to display (default: 10)",
    )
    args = parser.parse_args()

    predict(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )

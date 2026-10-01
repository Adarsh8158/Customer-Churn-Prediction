"""Flask API for churn prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"
CONFIG_PATH = ROOT / "model" / "model_config.json"

app = Flask(__name__)
CORS(app)

# Global model
pipeline = None


class CustomerData(BaseModel):
    """Pydantic model for customer input validation."""
    SeniorCitizen: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


def load_model() -> None:
    """Load the trained model."""
    global pipeline
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first."
        )
    pipeline = joblib.load(MODEL_PATH)


@app.before_request
def setup() -> None:
    """Initialize model on first request."""
    global pipeline
    if pipeline is None:
        load_model()


@app.route("/", methods=["GET"])
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Churn Prediction API is running",
    }


@app.route("/predict", methods=["POST"])
def predict() -> dict:
    """Predict churn for a single customer.
    
    Expected JSON body:
    {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "gender": "Male",
        ... (all customer features)
    }
    """
    try:
        data = request.get_json()
        customer = CustomerData(**data)
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Request error", "details": str(e)}), 400

    try:
        # Convert to DataFrame format expected by pipeline
        import pandas as pd
        df = pd.DataFrame([customer.dict()])
        
        prediction = pipeline.predict(df)[0]
        probability = float(pipeline.predict_proba(df)[0, 1])
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return jsonify({
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": probability,
            "risk_level": risk_level,
        })
    except Exception as e:
        return jsonify({"error": "Prediction error", "details": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict() -> dict:
    """Predict churn for multiple customers.
    
    Expected JSON body:
    {
        "customers": [
            {"SeniorCitizen": 0, "tenure": 12, ...},
            {"SeniorCitizen": 1, "tenure": 24, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        customers_data = data.get("customers", [])
        
        if not customers_data:
            return jsonify({"error": "No customers provided"}), 400
        
        # Validate each customer
        customers = [CustomerData(**c) for c in customers_data]
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "Request error", "details": str(e)}), 400

    try:
        import pandas as pd
        df = pd.DataFrame([c.dict() for c in customers])
        
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[:, 1]
        
        results = []
        for pred, prob in zip(predictions, probabilities):
            if prob < 0.3:
                risk = "Low"
            elif prob < 0.6:
                risk = "Medium"
            else:
                risk = "High"
            
            results.append({
                "prediction": "Churn" if pred == 1 else "No Churn",
                "churn_probability": float(prob),
                "risk_level": risk,
            })
        
        return jsonify({
            "total": len(results),
            "predictions": results,
        })
    except Exception as e:
        return jsonify({"error": "Prediction error", "details": str(e)}), 500


@app.errorhandler(404)
def not_found(error) -> dict:
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error) -> dict:
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

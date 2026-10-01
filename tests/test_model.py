"""Unit tests for the churn prediction model."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"
METRICS_PATH = ROOT / "model" / "training_metrics.json"


class TestModel:
    """Test suite for the trained model."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Load the trained model."""
        if not MODEL_PATH.exists():
            pytest.skip("Model not trained yet. Run 'python train_model.py' first.")
        return joblib.load(MODEL_PATH)

    @pytest.fixture(scope="class")
    def metrics(self):
        """Load training metrics."""
        if not METRICS_PATH.exists():
            pytest.skip("Metrics not found. Run 'python train_model.py' first.")
        with open(METRICS_PATH) as f:
            return json.load(f)

    def test_model_exists(self):
        """Test that model file exists."""
        assert MODEL_PATH.exists(), f"Model not found at {MODEL_PATH}"

    def test_metrics_exist(self):
        """Test that metrics file exists."""
        assert METRICS_PATH.exists(), f"Metrics not found at {METRICS_PATH}"

    def test_model_structure(self, pipeline):
        """Test that pipeline has correct structure."""
        assert hasattr(pipeline, "predict"), "Pipeline missing predict method"
        assert hasattr(pipeline, "predict_proba"), "Pipeline missing predict_proba method"
        assert pipeline.named_steps.get("preprocessor") is not None, "Missing preprocessor"
        assert pipeline.named_steps.get("model") is not None, "Missing model"

    def test_model_performance_accuracy(self, metrics):
        """Test that model accuracy is above threshold."""
        accuracy = metrics["test_metrics"]["accuracy"]
        assert accuracy > 0.75, f"Accuracy {accuracy} is below 0.75"

    def test_model_performance_roc_auc(self, metrics):
        """Test that model ROC-AUC is above threshold."""
        roc_auc = metrics["test_metrics"]["roc_auc"]
        assert roc_auc > 0.80, f"ROC-AUC {roc_auc} is below 0.80"

    def test_model_performance_recall(self, metrics):
        """Test that churn recall (true positive rate) is reasonable."""
        recall = metrics["test_metrics"]["classification_report"]["1"]["recall"]
        assert recall > 0.45, f"Recall {recall} is below 0.45 (too many false negatives)"

    def test_cv_stability(self, metrics):
        """Test that cross-validation scores are stable."""
        cv_mean = metrics["cross_validation_roc_auc_mean"]
        cv_std = metrics["cross_validation_roc_auc_std"]
        # Standard deviation should be low (model stable across folds)
        assert cv_std < 0.05, f"CV std dev {cv_std} is too high (model not stable)"
        # CV mean should be close to test performance
        test_roc = metrics["test_metrics"]["roc_auc"]
        assert abs(cv_mean - test_roc) < 0.05, f"CV mean {cv_mean} differs too much from test {test_roc}"

    def test_model_prediction_shape(self, pipeline):
        """Test that model produces correct output shape."""
        # Create a simple test case
        test_data = pd.DataFrame({
            "SeniorCitizen": [0],
            "tenure": [12],
            "MonthlyCharges": [65.5],
            "TotalCharges": [786.0],
            "gender": ["Male"],
            "Partner": ["Yes"],
            "Dependents": ["No"],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "InternetService": ["DSL"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["No"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["No"],
            "StreamingMovies": ["No"],
            "Contract": ["Month-to-month"],
            "PaperlessBilling": ["Yes"],
            "PaymentMethod": ["Electronic check"],
        })

        predictions = pipeline.predict(test_data)
        probas = pipeline.predict_proba(test_data)

        assert len(predictions) == 1, "Should return 1 prediction"
        assert probas.shape == (1, 2), "Should return (1, 2) probabilities"
        assert 0 <= probas[0, 1] <= 1, "Probability should be between 0 and 1"


class TestDataPipeline:
    """Test suite for data loading and preprocessing."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Load the raw dataset."""
        data_path = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not data_path.exists():
            pytest.skip(f"Dataset not found at {data_path}")
        return pd.read_csv(data_path)

    def test_dataset_exists(self):
        """Test that dataset file exists."""
        data_path = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        assert data_path.exists(), f"Dataset not found at {data_path}"

    def test_dataset_shape(self, dataset):
        """Test dataset has expected dimensions."""
        assert dataset.shape[0] > 5000, "Dataset should have at least 5000 rows"
        assert dataset.shape[1] >= 20, "Dataset should have at least 20 columns"

    def test_dataset_no_duplicates(self, dataset):
        """Test that dataset has no full duplicates."""
        duplicates = dataset.duplicated().sum()
        assert duplicates == 0, f"Dataset has {duplicates} duplicate rows"

    def test_churn_column_exists(self, dataset):
        """Test that Churn target column exists."""
        assert "Churn" in dataset.columns, "Churn column not found"
        assert dataset["Churn"].isin(["Yes", "No"]).all(), "Churn should only contain Yes/No"

import joblib
import pandas as pd

from api.config.settings import MODEL_PATH, THRESHOLD_PATH, MODEL_NAME
from api.services.feature_engineering import engineer_features


def load_artifacts():
    """Load the fitted preprocessing + XGBoost pipeline and threshold."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Decision threshold not found: {THRESHOLD_PATH}")

    fitted_pipeline = joblib.load(MODEL_PATH)
    decision_threshold = float(joblib.load(THRESHOLD_PATH))

    return fitted_pipeline, decision_threshold


model, threshold = load_artifacts()


def predict_churn(features: dict) -> dict:
    """Engineer raw customer features and run the fitted ML pipeline."""

    input_df = pd.DataFrame([features])
    input_df = engineer_features(input_df)

    # The saved artifact is a complete sklearn Pipeline. It expects the
    # 21 raw + engineered columns, not the already-transformed 50 columns.
    expected_features = list(model.feature_names_in_)
    input_df = input_df[expected_features]

    churn_probability = float(model.predict_proba(input_df)[0, 1])
    churn_prediction = int(churn_probability >= threshold)

    return {
        "prediction": "Churn" if churn_prediction == 1 else "Not Churn",
        "churn_probability": round(churn_probability, 4),
        "threshold": round(threshold, 4),
        "model": MODEL_NAME,
    }

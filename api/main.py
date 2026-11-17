from fastapi import FastAPI, HTTPException

from api.config.settings import MODEL_NAME, MODEL_VERSION
from api.schema.user_input import CustomerData
from api.schema.prediction_response import PredictionResponse
from api.services.predict import model, threshold, predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn using a tuned XGBoost pipeline",
    version=MODEL_VERSION,
)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running!",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": model is not None,
        "threshold": threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_customer_churn(customer: CustomerData):
    try:
        features = customer.features.model_dump()
        return predict_churn(features)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc

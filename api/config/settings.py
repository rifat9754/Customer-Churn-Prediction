from pathlib import Path

# Project root: customer-churn-prediction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "customer_churn_xgboost.pkl"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.pkl"

MODEL_NAME = "XGBoost"
MODEL_VERSION = "1.0.0"

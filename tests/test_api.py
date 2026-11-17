
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# Sample customer input
customer_payload = {
    "features": {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
        "gender": "Female",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
}


def test_home_endpoint():

    response = client.get("/")

    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint():

    response = client.post(
        "/predict",
        json=customer_payload
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Churn", "Not Churn"]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["threshold"] <= 1
    assert result["model"] == "XGBoost"


def test_invalid_tenure():

    invalid_payload = {
        "features": {
            **customer_payload["features"],
            "tenure": 100
        }
    }

    response = client.post(
        "/predict",
        json=invalid_payload
    )

    assert response.status_code == 422


def test_missing_feature():

    invalid_payload = {
        "features": {
            key: value
            for key, value in customer_payload["features"].items()
            if key != "gender"
        }
    }

    response = client.post(
        "/predict",
        json=invalid_payload
    )

    assert response.status_code == 422

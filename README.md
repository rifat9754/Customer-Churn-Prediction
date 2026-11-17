# Customer Churn Prediction

An end-to-end machine learning application that estimates whether a telecom customer is likely to churn. The project combines exploratory data analysis, feature engineering, model comparison and tuning, a trained XGBoost pipeline, a FastAPI prediction service, a Streamlit dashboard, and Docker-based backend packaging.

> **Project status:** ML training, model artifacts, FastAPI endpoints, automated API tests, Streamlit frontend, and a backend Dockerfile are present in the project. Public deployment URLs are not included here; add them after deploying and verifying the services.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Models Evaluated](#models-evaluated)
- [Final Model and Test Results](#final-model-and-test-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Run Locally](#run-locally)
- [FastAPI Endpoints](#fastapi-endpoints)
- [Docker](#docker)
- [Deployment](#deployment)
- [Testing](#testing)
- [Model Artifacts and Git](#model-artifacts-and-git)
- [Limitations](#limitations)

## Overview

Customer churn occurs when a customer stops using a company's products or services. Identifying customers who are at risk of leaving can help a business prioritize retention activities.

This application accepts customer account and service information and returns:

- A predicted class: `Churn` or `Not Churn`
- The estimated churn probability
- The decision threshold used for classification
- The model name

The Streamlit interface communicates with the FastAPI backend. The backend validates the submitted data, creates the engineered features, and passes the resulting row to the saved preprocessing-and-model pipeline.

## Architecture

```text
                  Customer / User
                         |
                         v
              Streamlit Dashboard
                  (Frontend)
                         |
                  HTTP Request
                         |
                         v
                FastAPI Application
                    (Backend)
                         |
                Pydantic Validation
                         |
                         v
                 Feature Engineering
                  - TenureGroup
                  - TotalServices
                         |
                         v
             Saved Scikit-learn Pipeline
                - Preprocessing
                - XGBoost Classifier
                         |
                         v
                Churn Prediction
                         |
                         v
              Prediction + Probability
```

The Dockerfile packages the **FastAPI backend and model artifacts**. The Streamlit frontend is a separate application and should be deployed separately, configured with the public FastAPI base URL.

## Dataset

- **Dataset:** IBM Telco Customer Churn
- **Source:** [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043
- **Columns:** 21
- **Target:** `Churn`
- **Class distribution:** approximately 73.46% `No` and 26.54% `Yes`

The raw dataset is not included in this repository by default. Download it from Kaggle and place the CSV at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Do not commit private, licensed, or otherwise restricted data. Check the dataset's terms before redistributing it.

## Machine Learning Workflow

The notebooks document the project workflow:

1. **Exploratory Data Analysis** – inspect distributions, missing values, class balance, and relationships between customer attributes and churn.
2. **Data Cleaning** – convert `TotalCharges` to numeric; whitespace-only values are handled as missing and filled with `0` for customers with zero tenure.
3. **Train/validation/test split** – use stratified splits to preserve the target distribution.
4. **Feature engineering**:
   - `TenureGroup`: groups customer tenure into `0-12`, `13-24`, `25-48`, and `49-72` month ranges.
   - `TotalServices`: counts subscribed phone/internet services using the same logic in training and API inference.
5. **Preprocessing** – scale numerical columns and one-hot encode categorical columns using a `ColumnTransformer`.
6. **Model comparison** – evaluate multiple classification algorithms using validation metrics, including Average Precision (PR-AUC).
7. **Cross-validation and hyperparameter tuning** – use cross-validation and randomized search for selected models.
8. **Threshold optimization** – choose a decision threshold using validation predictions rather than relying only on the default `0.5`.
9. **Final evaluation** – evaluate the selected model once on the held-out test set.
10. **Model serialization** – save the fitted pipeline and decision threshold for API inference.

Preprocessing is fitted on training data and then applied to validation/test data to avoid fitting transformations on held-out samples.

## Models Evaluated

Eight classification algorithms were compared:

| Model | Validation PR-AUC (Average Precision) |
|---|---:|
| XGBoost | 0.6426 |
| Gradient Boosting | 0.6404 |
| Random Forest | 0.6395 |
| Logistic Regression | 0.6393 |
| Support Vector Machine (SVM) | 0.6062 |
| Naive Bayes | 0.5802 |
| Decision Tree | 0.5779 |
| K-Nearest Neighbors (KNN) | 0.5195 |

The project prioritizes **PR-AUC / Average Precision** because the churn class is the minority class. Recall is also important because missing a customer who is likely to churn can reduce the usefulness of a retention workflow.

XGBoost and Gradient Boosting were among the strongest candidates and were further tuned. XGBoost was selected for the final inference pipeline after considering validation performance and the desired recall/threshold trade-off.

## Final Model and Test Results

**Selected model:** XGBoost  
**Decision threshold:** `0.536`  
**Evaluation split:** held-out test set

| Metric | Test score |
|---|---:|
| Accuracy | 75.97% |
| Precision | 53.16% |
| Recall | 78.21% |
| F1-score | 63.29% |
| ROC-AUC | 85.51% |
| PR-AUC (Average Precision) | 67.29% |

A customer is classified as `Churn` when the predicted churn probability is greater than or equal to `0.536`; otherwise, the prediction is `Not Churn`.

These are experimental results on the supplied dataset, not a guarantee of performance on new business data. The threshold reflects a precision/recall trade-off and may need recalibration for a different business objective or population.

## Project Structure

```text
customer-churn-prediction/
├── api/
│   ├── config/
│   │   └── settings.py
│   ├── schema/
│   │   ├── user_input.py
│   │   └── prediction_response.py
│   ├── services/
│   │   ├── feature_engineering.py
│   │   └── predict.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── models/
│   ├── customer_churn_xgboost.pkl
│   └── decision_threshold.pkl
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.txt
└── requirements-api.txt
```

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Machine learning | Scikit-learn, XGBoost |
| Model persistence | Joblib |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| API testing | Pytest, FastAPI TestClient, HTTPX |
| Containerization | Docker |

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/rifat9754/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

Replace the repository URL with your actual GitHub URL.

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-api.txt
```

For notebook development and testing, install the development dependencies as well:

```bash
python -m pip install -r requirements.txt
```

### 4. Start the FastAPI backend

Run this command from the repository root:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Useful local URLs:

- API root: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

### 5. Start the Streamlit frontend

Open a second terminal, activate the same virtual environment, and run from the repository root:

```bash
python -m pip install -r frontend/requirements.txt
python -m streamlit run frontend/app.py
```

Open http://localhost:8501.

In the Streamlit sidebar, set the FastAPI Base URL to:

```text
http://127.0.0.1:8000
```

Use **Check API connection** to verify that the frontend can reach the backend.

## FastAPI Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information and model version |
| GET | `/health` | Service and model-loaded status |
| POST | `/predict` | Predict customer churn |

### Example prediction request

```json
{
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
```

Send the request to:

```text
POST http://127.0.0.1:8000/predict
Content-Type: application/json
```

### Example response

```json
{
  "prediction": "Churn",
  "churn_probability": 0.72,
  "threshold": 0.536,
  "model": "XGBoost"
}
```

The probability above is illustrative; actual values depend on the submitted customer features.

The API accepts the 19 original customer features. It generates `TenureGroup` and `TotalServices` internally so the inference feature engineering matches the training workflow.

## Docker

The root `Dockerfile` builds the **FastAPI backend image**, installs the pinned API dependencies, copies the `api/` package and `models/` directory, and starts Uvicorn on port `8000`.

### 1. Build the image

Run from the repository root:

```bash
docker build -t customer-churn-api:latest .
```

### 2. Run the container

```bash
docker run --name customer-churn-api -p 8000:8000 customer-churn-api:latest
```

### 3. Verify the container

Open:

```text
http://localhost:8000/health
```

Or use:

```bash
curl http://localhost:8000/health
```

To stop and remove the container:

```bash
docker stop customer-churn-api
docker rm customer-churn-api
```

### 4. Push the backend image to Docker Hub

Log in:

```bash
docker login
```

Tag the image (replace `<dockerhub-username>`):

```bash
docker tag customer-churn-api:latest <dockerhub-username>/customer-churn-api:latest
```

Push:

```bash
docker push <dockerhub-username>/customer-churn-api:latest
```

The Docker image contains the backend and model files; it does not contain or serve the Streamlit frontend.

## Deployment

Deploy the backend and frontend as separate services:

### Backend – FastAPI

1. Push the project source and required model artifacts to GitHub.
2. Create a Docker-based web service on a container hosting platform (for example, Railway or Render).
3. Build from the repository root using the included `Dockerfile`.
4. Configure the service to expose/listen on port `8000` according to the hosting provider's port settings.
5. After deployment, verify the public `/health` endpoint and `/docs` page.

Example backend URL format:

```text
https://<your-backend-service-domain>
```

### Frontend – Streamlit

1. Deploy `frontend/app.py` as a Streamlit application (for example, using Streamlit Community Cloud).
2. Set the app's main file path to `frontend/app.py`.
3. Configure the `API_BASE_URL` environment variable to the deployed FastAPI base URL, without a trailing slash; alternatively, enter the URL in the app sidebar.
4. Verify the sidebar API connection and submit a test prediction.

Example environment variable:

```text
API_BASE_URL=https://<your-backend-service-domain>
```

### Deployment checklist

- [ ] Backend service is running.
- [ ] `GET /health` returns a healthy status and `model_loaded: true`.
- [ ] `GET /docs` is publicly reachable.
- [ ] Frontend is deployed.
- [ ] Frontend points to the correct public backend URL.
- [ ] A prediction request succeeds from the deployed frontend.
- [ ] No secrets, credentials, or private environment files are committed.

Add your actual live URLs here after verification:

- **Live frontend:** `https://<your-streamlit-app-url>`
- **Live API:** `https://<your-backend-service-domain>`
- **API documentation:** `https://<your-backend-service-domain>/docs`

## Testing

Run the API test suite from the repository root:

```bash
python -m pytest -v
```

The tests cover:

- Root endpoint response
- Health endpoint and model loading
- Successful prediction
- Invalid tenure validation
- Missing required feature validation

Using `python -m pytest` helps ensure pytest runs under the currently activated Python environment.

## Model Artifacts and Git

The backend requires these files at build/runtime:

```text
models/customer_churn_xgboost.pkl
models/decision_threshold.pkl
```

The first file is the complete fitted preprocessing + XGBoost pipeline. The second stores the decision threshold (`0.536`). The API does not need to load `preprocessor.pkl` separately.

**Important:** the current `.gitignore` pattern `models/*.pkl` ignores these required model artifacts. If you want a fresh GitHub clone to build the Docker image, update `.gitignore` so the two required files are tracked, or arrange another artifact download step during deployment.

For this project's current small model artifacts, you can replace:

```gitignore
# Model files
models/*.pkl
models/*.joblib
```

with:

```gitignore
# Ignore model artifacts by default, but track the files required by the API
models/*.pkl
models/*.joblib
!models/customer_churn_xgboost.pkl
!models/decision_threshold.pkl
```

Then check what Git will include:

```bash
git status --short
git check-ignore -v models/customer_churn_xgboost.pkl
```

If a model file was already tracked as ignored or the ignore rules are changed, stage the required artifacts explicitly:

```bash
git add -f models/customer_churn_xgboost.pkl models/decision_threshold.pkl
```

Do not commit `.venv/`, `.env`, `.git/`, `__pycache__/`, `.pytest_cache/`, or unnecessary raw/processed datasets.

## Limitations

- This is a portfolio and demonstration project, not a production churn-management system.
- Predictions are based on the IBM Telco dataset and may not generalize to another telecom provider without validation.
- The dashboard's overview metrics are static experiment results; they are not live business KPIs from a customer database.
- Churn probability is a model estimate, not a certainty.
- The selected decision threshold is specific to the validation workflow and may need recalibration for a different retention strategy.

## Author

**Your Name**

- GitHub: `https://github.com/rifat9754`
- LinkedIn: `https://www.linkedin.com/in/ahanaf-tahmid-rifat-220663345/`

---

If you use or extend this project, please review the dataset's terms and document any changes to the training data, model, threshold, or deployment configuration.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """The 19 original Telco customer features accepted by the API."""

    model_config = ConfigDict(extra="forbid")

    SeniorCitizen: Literal[0, 1] = Field(description="Whether the customer is a senior citizen (0 or 1)")
    tenure: int = Field(ge=0, le=72, description="Number of months the customer has stayed")
    MonthlyCharges: float = Field(ge=0, description="Monthly amount charged")
    TotalCharges: float = Field(ge=0, description="Total amount charged")

    gender: Literal["Female", "Male"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class CustomerData(BaseModel):
    features: CustomerFeatures

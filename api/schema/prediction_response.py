from pydantic import BaseModel, Field
from typing import Literal


class PredictionResponse(BaseModel):
    prediction: Literal["Churn", "Not Churn"] = Field(
        description="Predicted customer churn class"
    )
    churn_probability: float = Field(
        ge=0,
        le=1,
        description="Estimated probability that the customer will churn"
    )
    threshold: float = Field(
        ge=0,
        le=1,
        description="Decision threshold applied to churn probability"
    )
    model: str = Field(description="Name of the model used for prediction")

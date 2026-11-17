import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature engineering used during model training."""

    df = data.copy()

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49-72 Months",
        ],
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    df["TotalServices"] = (
        df[service_columns].eq("Yes").sum(axis=1)
        + df["InternetService"].ne("No").astype(int)
    )

    return df

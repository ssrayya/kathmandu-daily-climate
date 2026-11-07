
from pathlib import Path

import numpy as np
import pandas as pd


# Dataset location
DATA_PATH = Path("data/Kathmandu_Daily_Climate_Data.csv")


# Climate variables used as input features
WEATHER_COLUMNS = [
    "T2M",
    "T2MWET",
    "TS",
    "T2M_RANGE",
    "T2M_MAX",
    "T2M_MIN",
    "QV2M",
    "RH2M",
    "PRECTOTCORR",
    "PS",
    "WS2M",
    "WS2M_MAX",
    "WS2M_MIN",
    "WS2M_RANGE",
    "WS10M",
    "WS10M_MAX",
    "WS10M_MIN",
    "WS10M_RANGE",
]


def load_and_prepare_data():
    """Load climate data and prepare next-day prediction features."""

    df = pd.read_csv(DATA_PATH)

    # Construct the date column
    df["DATE"] = pd.to_datetime(
        df["YEAR"].astype(str)
        + df["DOY"].astype(str).str.zfill(3),
        format="%Y%j"
    )

    # Sort observations chronologically
    df = df.sort_values("DATE").reset_index(drop=True)

    # Create a copy of the weather observations
    X = df[WEATHER_COLUMNS].copy()

    # Add previous temperature observations
    X["T2M_LAG_1"] = df["T2M"]
    X["T2M_LAG_2"] = df["T2M"].shift(1)
    X["T2M_LAG_3"] = df["T2M"].shift(2)

    # Add rolling temperature averages
    X["T2M_ROLLING_3"] = df["T2M"].rolling(3).mean()
    X["T2M_ROLLING_7"] = df["T2M"].rolling(7).mean()

    # Predict the following day's average temperature
    y = df["T2M"].shift(-1)

    # Add the prediction date's known seasonal information
    prediction_dates = df["DATE"].shift(-1)
    prediction_doy = prediction_dates.dt.day_of_year

    X["NEXT_DAY_DOY_SIN"] = np.sin(
        2 * np.pi * prediction_doy / 365.25
    )
    X["NEXT_DAY_DOY_COS"] = np.cos(
        2 * np.pi * prediction_doy / 365.25
    )

    # Remove rows with missing lag features or target values
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)
    prediction_dates = prediction_dates.loc[valid_rows].reset_index(drop=True)

    return X, y, prediction_dates



def get_train_validation_test_splits(X, y):

    total_rows = len(X)

    train_end = int(total_rows * 0.70)
    validation_end = int(total_rows * 0.85)

    # Leave one observation between each split
    X_train = X.iloc[:train_end - 1]
    X_val = X.iloc[train_end:validation_end - 1]
    X_test = X.iloc[validation_end:]

    y_train = y.iloc[:train_end - 1]
    y_val = y.iloc[train_end:validation_end - 1]
    y_test = y.iloc[validation_end:]

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )



if __name__ == "__main__":

    X, y, prediction_dates = load_and_prepare_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = get_train_validation_test_splits(X, y)

    print("Dataset preprocessing completed!")

    print("\nFeature shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nTraining records:", len(X_train))
    print("Validation records:", len(X_val))
    print("Testing records:", len(X_test))

    print("\nPrediction target:")
    print("Next-day average temperature (T2M)")

    print("\nFirst 5 prediction dates:")
    print(prediction_dates.head())

    print("\nFirst 5 target values:")
    print(y.head())

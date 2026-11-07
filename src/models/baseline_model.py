
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocess import (
    load_and_prepare_data,
    get_train_validation_test_splits
)


def main():
    # Load prepared dataset
    X, y, prediction_dates = load_and_prepare_data()

    # Get test data
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = get_train_validation_test_splits(X, y)

    # Persistence baseline:
    # Tomorrow's temperature = today's temperature
    y_pred = X_test["T2M"].reset_index(drop=True)

    y_actual = y_test.reset_index(drop=True)

    # Calculate evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(y_actual, y_pred),
        "MSE": mean_squared_error(y_actual, y_pred),
        "RMSE": mean_squared_error(y_actual, y_pred) ** 0.5,
        "R2": r2_score(y_actual, y_pred)
    }

    print("\nPersistence Baseline Results")
    print("----------------------------")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save results
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        results_dir / "baseline_results.csv",
        index=False
    )

    print("\nBaseline results saved successfully!")


if __name__ == "__main__":
    main()


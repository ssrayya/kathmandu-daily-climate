
import json
from pathlib import Path

import joblib
import pandas as pd
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
    models_dir = Path("outputs/models")
    results_dir = Path("outputs/results")
    figures_dir = Path("outputs/figures")

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load the prepared dataset
    X, y, prediction_dates = load_and_prepare_data()

    # Get the same chronological data splits
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = get_train_validation_test_splits(X, y)

    models = {
        "Random Forest": "random_forest_best.joblib",
        "Ridge": "ridge_best.joblib",
        "Lasso": "lasso_best.joblib"
    }

    results = []

    print("\nFinal Test Evaluation")
    print("---------------------")

    for model_name, filename in models.items():

        model = joblib.load(models_dir / filename)

        # Predict using unseen test data
        y_test_pred = model.predict(X_test)

        # Calculate evaluation metrics
        metrics = {
            "Model": model_name,
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": mean_squared_error(y_test, y_test_pred) ** 0.5,
            "R2": r2_score(y_test, y_test_pred)
        }

        results.append(metrics)

        print(f"\n{model_name}")
        for metric, value in metrics.items():
            if metric != "Model":
                print(f"{metric}: {value:.4f}")

    # Create comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(
        "RMSE"
    ).reset_index(drop=True)

    # Save test comparison
    comparison_df.to_csv(
        results_dir / "final_test_comparison.csv",
        index=False
    )

    # Identify the best model based on test RMSE
    best_model = comparison_df.iloc[0]

    print("\nBest Model Based on Test RMSE:")
    print(
        f"{best_model['Model']} "
        f"(RMSE: {best_model['RMSE']:.4f} °C)"
    )

    # Save final results as JSON
    final_results = {
        "best_model": best_model["Model"],
        "best_test_rmse": float(best_model["RMSE"]),
        "test_metrics": results
    }

    with open(
        results_dir / "final_test_results.json",
        "w"
    ) as file:
        json.dump(final_results, file, indent=4)

    print("\nFinal test results saved successfully!")


if __name__ == "__main__":
    main()

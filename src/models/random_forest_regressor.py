
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.preprocess import (
    load_and_prepare_data,
    get_train_validation_test_splits
)


# Reproducibility
RANDOM_STATE = 42


def main():

    # Create output directories
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    # Load prepared data
    X, y, prediction_dates = load_and_prepare_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = get_train_validation_test_splits(X, y)

    print("Training Random Forest Regressor...")

    # Initialize model
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 15],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 1.0]
    }

    # Time-series cross-validation
    time_cv = TimeSeriesSplit(n_splits=5, gap=1)

    # Grid search using training data only
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=time_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Validation predictions
    y_val_pred = best_model.predict(X_val)

    # Validation metrics
    metrics = {
        "MAE": mean_absolute_error(y_val, y_val_pred),
        "MSE": mean_squared_error(y_val, y_val_pred),
        "RMSE": mean_squared_error(
            y_val, y_val_pred
        ) ** 0.5,
        "R2": r2_score(y_val, y_val_pred)
    }

    # Save best model
    joblib.dump(
        best_model,
        "outputs/models/random_forest_best.joblib"
    )

    # Save hyperparameter search history
    cv_results = pd.DataFrame(grid_search.cv_results_)

    cv_results.to_csv(
        "outputs/results/random_forest_search_history.csv",
        index=False
    )

    # Save validation metrics and best parameters
    results = {
        "best_parameters": grid_search.best_params_,
        "best_cv_rmse": -grid_search.best_score_,
        "validation_metrics": metrics
    }

    with open(
        "outputs/results/random_forest_results.json",
        "w"
    ) as file:
        json.dump(results, file, indent=4)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation RMSE:")
    print(-grid_search.best_score_)

    print("\nValidation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nRandom Forest training completed!")
    print("Model and results saved successfully.")


if __name__ == "__main__":
    main()

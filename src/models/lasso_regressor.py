
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

RANDOM_STATE = 42


def main():
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    X, y, prediction_dates = load_and_prepare_data()

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) = get_train_validation_test_splits(X, y)

    print("Training Lasso Regression...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000))
    ])

    param_grid = {
        "model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]
    }

    time_cv = TimeSeriesSplit(n_splits=5, gap=1)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=time_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_val_pred = best_model.predict(X_val)

    metrics = {
        "MAE": mean_absolute_error(y_val, y_val_pred),
        "MSE": mean_squared_error(y_val, y_val_pred),
        "RMSE": mean_squared_error(y_val, y_val_pred) ** 0.5,
        "R2": r2_score(y_val, y_val_pred)
    }

    joblib.dump(
        best_model,
        "outputs/models/lasso_best.joblib"
    )

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(
        "outputs/results/lasso_search_history.csv",
        index=False
    )

    results = {
        "best_parameters": grid_search.best_params_,
        "best_cv_rmse": -grid_search.best_score_,
        "validation_metrics": metrics
    }

    with open("outputs/results/lasso_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation RMSE:")
    print(-grid_search.best_score_)

    print("\nValidation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nLasso Regression training completed!")
    print("Model and results saved successfully.")


if __name__ == "__main__":
    main()

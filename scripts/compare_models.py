
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("outputs/results")
FIGURES_DIR = Path("outputs/figures")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load machine learning test results
ml_results = pd.read_csv(
    RESULTS_DIR / "final_test_comparison.csv"
)

# Load persistence baseline results
baseline = pd.read_csv(
    RESULTS_DIR / "baseline_results.csv"
)

baseline["Model"] = "Persistence Baseline"

# Combine all four models
comparison = pd.concat(
    [ml_results, baseline],
    ignore_index=True
)

comparison = comparison[
    ["Model", "MAE", "MSE", "RMSE", "R2"]
]

comparison = comparison.sort_values(
    "RMSE"
).reset_index(drop=True)

# Save final comparison
comparison.to_csv(
    RESULTS_DIR / "final_model_comparison.csv",
    index=False
)

print("\nFinal Model Comparison:")
print(comparison.to_string(index=False))

# Plot RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(comparison["Model"], comparison["RMSE"])
plt.title("Final Test RMSE Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE (°C)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "final_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFinal comparison table and graph saved successfully!")

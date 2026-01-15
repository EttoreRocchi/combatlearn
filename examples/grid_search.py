"""
Grid Search Example
====================

This example demonstrates hyperparameter tuning with GridSearchCV,
including ComBat-specific parameters alongside classifier parameters.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from combatlearn import ComBat

# Set random seed
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=20,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

# Add batch effects
batches = np.repeat(["Batch_A", "Batch_B", "Batch_C"], [70, 70, 60])
for i, batch in enumerate(batches):
    if batch == "Batch_A":
        X[i] += 2.0
    elif batch == "Batch_B":
        X[i] -= 1.5

# Convert to DataFrame
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="target")
batch = pd.Series(batches, index=X.index, name="batch")

# Add covariates
age = pd.DataFrame({"age": np.random.normal(45, 10, len(X))}, index=X.index)

print("=" * 70)
print("Grid Search with ComBat")
print("=" * 70)
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.bincount(y)}")
print(f"Batches: {batch.value_counts().to_dict()}")

# Create pipeline
pipeline = Pipeline(
    [
        ("combat", ComBat(batch=batch, method="fortin", continuous_covariates=age)),
        ("scaler", StandardScaler()),
        ("classifier", SVC(random_state=42)),
    ]
)

# Define parameter grid
# Note: Combat parameters are prefixed with "combat__"
param_grid = {
    "combat__method": ["johnson", "fortin"],
    "combat__parametric": [True, False],
    "combat__mean_only": [True, False],
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__kernel": ["linear", "rbf"],
}

print("\n" + "=" * 70)
print("Parameter Grid")
print("=" * 70)
print("ComBat parameters:")
print(f"  - method: {param_grid['combat__method']}")
print(f"  - parametric: {param_grid['combat__parametric']}")
print(f"  - mean_only: {param_grid['combat__mean_only']}")
print("\nClassifier parameters:")
print(f"  - C: {param_grid['classifier__C']}")
print(f"  - kernel: {param_grid['classifier__kernel']}")

total_combinations = (
    len(param_grid["combat__method"])
    * len(param_grid["combat__parametric"])
    * len(param_grid["combat__mean_only"])
    * len(param_grid["classifier__C"])
    * len(param_grid["classifier__kernel"])
)
print(f"\nTotal parameter combinations: {total_combinations}")

# Perform grid search
print("\n" + "=" * 70)
print("Running Grid Search (this may take a moment)...")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X, y)

# Display results
print("\n" + "=" * 70)
print("Grid Search Results")
print("=" * 70)
print(f"\nBest score (accuracy): {grid_search.best_score_:.4f}")
print("\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# Show top 5 parameter combinations
print("\n" + "=" * 70)
print("Top 5 Parameter Combinations")
print("=" * 70)

results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values("rank_test_score")

for idx, (_i, row) in enumerate(results.head(5).iterrows(), 1):
    print(f"\n{idx}. Score: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
    print(f"   Parameters: {row['params']}")

# Analyze impact of ComBat parameters
print("\n" + "=" * 70)
print("Impact of ComBat Parameters")
print("=" * 70)

# Group by Combat method
method_scores = results.groupby("param_combat__method")["mean_test_score"].agg(["mean", "std"])
print("\nBy method:")
print(method_scores)

# Group by parametric
parametric_scores = results.groupby("param_combat__parametric")["mean_test_score"].agg(
    ["mean", "std"]
)
print("\nBy parametric:")
print(parametric_scores)

# Group by mean_only
mean_only_scores = results.groupby("param_combat__mean_only")["mean_test_score"].agg(
    ["mean", "std"]
)
print("\nBy mean_only:")
print(mean_only_scores)

print("\n" + "=" * 70)
print("Grid search completed successfully!")
print("=" * 70)

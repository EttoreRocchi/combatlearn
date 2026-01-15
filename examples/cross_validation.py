"""
Cross-Validation Example
=========================

This example demonstrates how to use ComBat within scikit-learn's
cross-validation framework to prevent data leakage.

The key insight: ComBat must be fitted on the training set only,
then applied to both training and test sets during each CV fold.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from combatlearn import ComBat

# Set random seed
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(
    n_samples=300,
    n_features=50,
    n_informative=30,
    n_redundant=10,
    n_classes=2,
    random_state=42,
)

# Add batch effects
batches = np.repeat(["Batch_1", "Batch_2", "Batch_3"], [100, 100, 100])
for i, batch in enumerate(batches):
    if batch == "Batch_1":
        X[i] += 1.5
    elif batch == "Batch_2":
        X[i] -= 1.0
    elif batch == "Batch_3":
        X[i] += 0.3

# Convert to DataFrame
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
y = pd.Series(y, name="target")
batch = pd.Series(batches, index=X.index, name="batch")

print("=" * 70)
print("Cross-Validation with ComBat")
print("=" * 70)
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes: {np.bincount(y)}")
print(f"Batches: {batch.value_counts().to_dict()}")

# Example 1: Without batch correction (baseline)
print("\n" + "=" * 70)
print("1. Baseline: No Batch Correction")
print("=" * 70)

pipeline_baseline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_baseline = cross_val_score(pipeline_baseline, X, y, cv=cv, scoring="roc_auc")

print(f"Cross-validated ROC-AUC scores: {scores_baseline}")
print(f"Mean ROC-AUC: {scores_baseline.mean():.4f} (+/- {scores_baseline.std():.4f})")

# Example 2: With ComBat batch correction
print("\n" + "=" * 70)
print("2. With ComBat Batch Correction (Johnson Method)")
print("=" * 70)

pipeline_combat = Pipeline(
    [
        ("combat", ComBat(batch=batch, method="johnson")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

scores_combat = cross_val_score(pipeline_combat, X, y, cv=cv, scoring="roc_auc")

print(f"Cross-validated ROC-AUC scores: {scores_combat}")
print(f"Mean ROC-AUC: {scores_combat.mean():.4f} (+/- {scores_combat.std():.4f})")

# Example 3: With ComBat + covariates
print("\n" + "=" * 70)
print("3. With ComBat + Covariates (Fortin Method)")
print("=" * 70)

# Add synthetic covariates
age = pd.DataFrame({"age": np.random.normal(50, 12, len(X))}, index=X.index)
site = pd.DataFrame({"site": np.random.choice(["Site_A", "Site_B"], len(X))}, index=X.index)

pipeline_combat_cov = Pipeline(
    [
        (
            "combat",
            ComBat(
                batch=batch,
                method="fortin",
                continuous_covariates=age,
                discrete_covariates=site,
            ),
        ),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]
)

scores_combat_cov = cross_val_score(pipeline_combat_cov, X, y, cv=cv, scoring="roc_auc")

print(f"Cross-validated ROC-AUC scores: {scores_combat_cov}")
print(f"Mean ROC-AUC: {scores_combat_cov.mean():.4f} (+/- {scores_combat_cov.std():.4f})")

# Comparison
print("\n" + "=" * 70)
print("Summary: Performance Comparison")
print("=" * 70)
print(
    f"Baseline (no correction):     {scores_baseline.mean():.4f} (+/- {scores_baseline.std():.4f})"
)
print(f"ComBat (Johnson):             {scores_combat.mean():.4f} (+/- {scores_combat.std():.4f})")
print(
    f"ComBat (Fortin w/ covariates):{scores_combat_cov.mean():.4f} (+/- {scores_combat_cov.std():.4f})"
)

improvement_johnson = (scores_combat.mean() - scores_baseline.mean()) / scores_baseline.mean() * 100
improvement_fortin = (
    (scores_combat_cov.mean() - scores_baseline.mean()) / scores_baseline.mean() * 100
)

print("\nImprovement over baseline:")
print(f"  Johnson method: {improvement_johnson:+.2f}%")
print(f"  Fortin method:  {improvement_fortin:+.2f}%")

print("\n" + "=" * 70)
print("Key Takeaway:")
print("ComBat is applied within each CV fold, preventing data leakage!")
print("The batch correction parameters are learned from training data only.")
print("=" * 70)

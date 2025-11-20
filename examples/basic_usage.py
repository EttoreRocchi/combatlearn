"""
Basic Usage Example
===================

This example demonstrates the basic usage of ComBat for batch effect correction
using all three available methods: Johnson, Fortin, and Chen.
"""

import numpy as np
import pandas as pd
from combatlearn import ComBat

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data with batch effects
n_samples = 150
n_features = 50
batches = np.repeat(["Batch_A", "Batch_B", "Batch_C"], [50, 50, 50])

# Create data with artificial batch effects
X = np.random.randn(n_samples, n_features)
for i, batch in enumerate(batches):
    if batch == "Batch_A":
        X[i] += 2.0  # Add batch effect
    elif batch == "Batch_B":
        X[i] -= 1.5  # Add different batch effect
    elif batch == "Batch_C":
        X[i] += 0.5  # Add smaller batch effect

# Convert to DataFrame
X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
batch = pd.Series(batches, index=X.index, name="batch")

print("=" * 60)
print("ComBat Basic Usage Example")
print("=" * 60)
print(f"\nData shape: {X.shape}")
print(f"Batches: {batch.value_counts().to_dict()}")

# Example 1: Johnson method (classic ComBat)
print("\n" + "=" * 60)
print("1. Johnson Method (Classic ComBat)")
print("=" * 60)

combat_johnson = ComBat(batch=batch, method="johnson")
X_corrected_johnson = combat_johnson.fit_transform(X)

print(f"Original data mean (Batch A): {X.loc[batch == 'Batch_A'].mean().mean():.3f}")
print(f"Corrected data mean (Batch A): {X_corrected_johnson.loc[batch == 'Batch_A'].mean().mean():.3f}")
print(f"Original data mean (Batch B): {X.loc[batch == 'Batch_B'].mean().mean():.3f}")
print(f"Corrected data mean (Batch B): {X_corrected_johnson.loc[batch == 'Batch_B'].mean().mean():.3f}")

# Example 2: Fortin method (with covariates)
print("\n" + "=" * 60)
print("2. Fortin Method (neuroCombat with Covariates)")
print("=" * 60)

# Add covariates
age = pd.DataFrame({
    "age": np.random.normal(50, 10, n_samples)
}, index=X.index)

diagnosis = pd.DataFrame({
    "diagnosis": np.random.choice(["healthy", "disease"], n_samples)
}, index=X.index)

combat_fortin = ComBat(
    batch=batch,
    method="fortin",
    continuous_covariates=age,
    discrete_covariates=diagnosis
)
X_corrected_fortin = combat_fortin.fit_transform(X)

print("Covariates included:")
print(f"  - Age (continuous): mean={age['age'].mean():.1f}, std={age['age'].std():.1f}")
print(f"  - Diagnosis (discrete): {diagnosis['diagnosis'].value_counts().to_dict()}")
print(f"\nCorrected data mean (all batches): {X_corrected_fortin.mean().mean():.3f}")

# Example 3: Chen method (CovBat - PCA-based)
print("\n" + "=" * 60)
print("3. Chen Method (CovBat - PCA-based)")
print("=" * 60)

combat_chen = ComBat(
    batch=batch,
    method="chen",
    continuous_covariates=age,
    discrete_covariates=diagnosis,
    covbat_cov_thresh=0.95  # Retain 95% of variance
)
X_corrected_chen = combat_chen.fit_transform(X)

n_components = combat_chen._model._covbat_n_pc
print(f"PCA components retained: {n_components}")
print(f"Corrected data mean (all batches): {X_corrected_chen.mean().mean():.3f}")

# Example 4: Using mean_only parameter
print("\n" + "=" * 60)
print("4. Mean-Only Correction")
print("=" * 60)

combat_mean_only = ComBat(batch=batch, method="johnson", mean_only=True)
X_corrected_mean_only = combat_mean_only.fit_transform(X)

print("Mean-only mode: Only batch mean is corrected, variance is preserved")
print(f"Original variance (Batch A): {X.loc[batch == 'Batch_A'].var().mean():.3f}")
print(f"Corrected variance (Batch A, mean_only=True): {X_corrected_mean_only.loc[batch == 'Batch_A'].var().mean():.3f}")
print(f"Corrected variance (Batch A, mean_only=False): {X_corrected_johnson.loc[batch == 'Batch_A'].var().mean():.3f}")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)

# Quick Start

This tutorial will get you started with combatlearn in just a few minutes.

## Basic Usage

### 1. Import and Prepare Data

```python
import pandas as pd
import numpy as np
from combatlearn import ComBat

# Create sample data with batch effects
np.random.seed(42)
n_samples = 150
n_features = 50

# Generate features
X = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"feature_{i}" for i in range(n_features)]
)

# Define batches
batch = pd.Series(
    ["Batch_A"] * 50 + ["Batch_B"] * 50 + ["Batch_C"] * 50,
    name="batch"
)

# Add artificial batch effects
X.loc[batch == "Batch_A"] += 2.0
X.loc[batch == "Batch_B"] -= 1.5
```

### 2. Apply ComBat Correction

```python
# Create and fit ComBat
combat = ComBat(batch=batch, method="johnson")
X_corrected = combat.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Corrected shape: {X_corrected.shape}")
```

### 3. Compare Before and After

```python
print("Mean by batch (before):")
print(X.groupby(batch).mean().mean(axis=1))

print("\nMean by batch (after):")
print(X_corrected.groupby(batch).mean().mean(axis=1))
```

## Using with Scikit-learn

combatlearn integrates seamlessly with scikit-learn pipelines:

### Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Create labels (for demonstration)
y = np.random.choice([0, 1], size=n_samples)

# Build pipeline
pipe = Pipeline([
    ("combat", ComBat(batch=batch, method="johnson")),
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf"))
])

# Cross-validation (ComBat is fitted on train folds only!)
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    "combat__method": ["johnson", "fortin"],
    "combat__parametric": [True, False],
    "classifier__C": [0.1, 1.0, 10.0]
}

# Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X, y)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
```

## Adding Covariates

For the Fortin and Chen methods, you can include covariates to preserve biological variation:

```python
# Create covariates
age = pd.DataFrame({"age": np.random.normal(50, 10, n_samples)})
sex = pd.DataFrame({"sex": np.random.choice(["M", "F"], n_samples)})

# Apply ComBat with covariates
combat = ComBat(
    batch=batch,
    method="fortin",
    continuous_covariates=age,
    discrete_covariates=sex
)

X_corrected = combat.fit_transform(X)
```

## Visualization

combatlearn includes built-in visualization tools:

```python
import matplotlib.pyplot as plt

# Fit ComBat
combat = ComBat(batch=batch, method="johnson").fit(X)

# Visualize transformation effect
fig = combat.plot_transformation(
    X,
    reduction_method='pca',
    n_components=2,
    plot_type='static'
)

plt.show()
```

## Method Selection

Choose the appropriate method for your use case:

| Method | Use When | Supports Covariates |
|--------|----------|---------------------|
| **Johnson** | Simple batch correction, no covariates | ❌ No |
| **Fortin** | Need to preserve covariate effects | ✅ Yes |
| **Chen** | High-dimensional data, feature-specific effects | ✅ Yes |

## Next Steps

- Learn more about [ComBat Methods](user-guide/methods.md)
- Explore [Parameters](user-guide/parameters.md)
- See detailed [Examples](examples/basic-usage.md)
- Check the [API Reference](api.md)

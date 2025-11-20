# Basic Usage Example

This example demonstrates the basic usage of all three ComBat methods.

See the full script: [`examples/basic_usage.py`](https://github.com/EttoreRocchi/combatlearn/blob/main/examples/basic_usage.py)

## Overview

This example shows:
- Johnson method (classic ComBat)
- Fortin method (with covariates)
- Chen method (PCA-based)
- Mean-only correction

## Running the Example

```bash
python examples/basic_usage.py
```

## Code Walkthrough

### 1. Generate Data

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 150
n_features = 50

# Create data
X = np.random.randn(n_samples, n_features)
batches = np.repeat(["Batch_A", "Batch_B", "Batch_C"], [50, 50, 50])

# Add batch effects
for i, batch in enumerate(batches):
    if batch == "Batch_A":
        X[i] += 2.0
    elif batch == "Batch_B":
        X[i] -= 1.5
```

### 2. Johnson Method

```python
from combatlearn import ComBat

combat_johnson = ComBat(batch=batch, method="johnson")
X_corrected = combat_johnson.fit_transform(X)

print(f"Original mean: {X.mean():.3f}")
print(f"Corrected mean: {X_corrected.mean():.3f}")
```

### 3. Fortin Method (with covariates)

```python
# Add covariates
age = pd.DataFrame({"age": np.random.normal(50, 10, n_samples)})
diagnosis = pd.DataFrame({"diagnosis": np.random.choice(["healthy", "disease"], n_samples)})

combat_fortin = ComBat(
    batch=batch,
    method="fortin",
    continuous_covariates=age,
    discrete_covariates=diagnosis
)
X_corrected = combat_fortin.fit_transform(X)
```

### 4. Chen Method (PCA-based)

```python
combat_chen = ComBat(
    batch=batch,
    method="chen",
    continuous_covariates=age,
    discrete_covariates=diagnosis,
    covbat_cov_thresh=0.95
)
X_corrected = combat_chen.fit_transform(X)

n_components = combat_chen._model._covbat_n_pc
print(f"PCA components: {n_components}")
```

## Next Steps

- Try [Grid Search Example](grid-search.md)
- Learn about [Cross-Validation](../user-guide/cross-validation.md)

# API Reference

Complete API documentation for combatlearn.

## ComBat

The main scikit-learn compatible transformer for batch effect correction.

::: combatlearn.ComBat
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
      members:
        - __init__
        - fit
        - transform
        - fit_transform
        - plot_transformation
        - compute_batch_metrics
        - metrics_

## Usage Examples

### Basic Usage

```python
from combatlearn import ComBat

combat = ComBat(batch=batch_labels, method="johnson")
X_corrected = combat.fit_transform(X)
```

### With Covariates

```python
combat = ComBat(
    batch=batch_labels,
    method="fortin",
    continuous_covariates=age_data,
    discrete_covariates=sex_data
)
X_corrected = combat.fit_transform(X)
```

### In Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ("combat", ComBat(batch=batch_labels)),
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Computing Batch Effect Metrics

```python
# Option 1: Auto-compute during fit_transform
combat = ComBat(batch=batch_labels, compute_metrics=True)
X_corrected = combat.fit_transform(X)

# Access cached metrics
metrics = combat.metrics_

# View batch effect metrics
for name, vals in metrics['batch_effect'].items():
    print(f"{name}: {vals['before']:.3f} -> {vals['after']:.3f}")
```

```python
# Option 2: On-demand computation for new data
combat = ComBat(batch=train_batch)
combat.fit(X_train)

# Compute metrics on test data with different batch labels
test_metrics = combat.compute_batch_metrics(
    X_test,
    batch=test_batch_labels,
    k_neighbors=[5, 10, 25]
)
```

## Method Parameters

### Johnson Method

```python
ComBat(
    batch=batch_labels,
    method="johnson",
    parametric=True,
    mean_only=False,
    reference_batch=None,
    eps=1e-8
)
```

### Fortin Method

```python
ComBat(
    batch=batch_labels,
    method="fortin",
    discrete_covariates=categorical_data,
    continuous_covariates=numerical_data,
    parametric=True,
    mean_only=False,
    reference_batch=None,
    eps=1e-8
)
```

### Chen Method

```python
ComBat(
    batch=batch_labels,
    method="chen",
    discrete_covariates=categorical_data,
    continuous_covariates=numerical_data,
    parametric=True,
    mean_only=False,
    reference_batch=None,
    covbat_cov_thresh=0.9,
    eps=1e-8
)
```

## Return Types

### fit()

Returns `self` for method chaining.

```python
combat = ComBat(batch=batch).fit(X)
```

### transform()

Returns `pd.DataFrame` with corrected data.

```python
X_corrected = combat.transform(X)
```

### fit_transform()

Returns `pd.DataFrame` with corrected data.

```python
X_corrected = combat.fit_transform(X)
```

### plot_transformation()

Returns matplotlib `Figure` or plotly `Figure` depending on `plot_type`.

Optionally returns tuple `(Figure, dict)` when `return_embeddings=True`.

```python
# Returns figure only
fig = combat.plot_transformation(X)

# Returns figure and embeddings
fig, embeddings = combat.plot_transformation(X, return_embeddings=True)
```

### compute_batch_metrics()

Returns `dict` with comprehensive batch effect metrics.

The returned dictionary contains:
- `batch_effect`: Silhouette, Davies-Bouldin, kBET, LISI, variance ratio (each with `before` and `after` values)
- `preservation`: k-NN preservation, distance correlation
- `alignment`: Centroid distance, Levene statistic

```python
# Compute metrics on fitted model
metrics = combat.compute_batch_metrics(X)

# View batch effect metrics
for name, vals in metrics['batch_effect'].items():
    print(f"{name}: {vals['before']:.3f} -> {vals['after']:.3f}")

# Compute metrics on new data with different batch labels
test_metrics = combat.compute_batch_metrics(X_test, batch=test_batch_labels)
```

### metrics_ property

Returns `Optional[dict]` - cached metrics from last `fit_transform()` when `compute_metrics=True`.

```python
combat = ComBat(batch=batch, compute_metrics=True)
X_corrected = combat.fit_transform(X)
print(combat.metrics_)  # Access cached metrics
```

## Exceptions

### ValueError

Raised when:
- Invalid method specified
- Invalid parameter values
- Batch has fewer than 2 samples
- Reference batch not found in data
- Mismatched indices between X and batch/covariates
- Transform called on unfitted model
- Unseen batch levels during transform

```python
# Example: invalid method
try:
    combat = ComBat(batch=batch, method="invalid")
except ValueError as e:
    print(e)  # "method must be 'johnson', 'fortin', or 'chen'"
```

### TypeError

Raised when:
- Invalid type for covbat_cov_thresh
- Invalid parameter types

```python
# Example: wrong type
try:
    combat = ComBat(batch=batch, method="chen", covbat_cov_thresh="invalid")
except TypeError as e:
    print(e)  # "covbat_cov_thresh must be float or int"
```

## Type Hints

combatlearn uses type hints for better IDE support:

```python
from typing import Optional
import pandas as pd
from combatlearn import ComBat

def correct_batches(
    data: pd.DataFrame,
    batch_labels: pd.Series,
    age: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    combat = ComBat(
        batch=batch_labels,
        method="fortin",
        continuous_covariates=age
    )
    return combat.fit_transform(data)
```

## See Also

- [User Guide](user-guide/overview.md)
- [Examples](examples/basic-usage.md)
- [Parameters Reference](user-guide/parameters.md)

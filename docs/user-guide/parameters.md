# Parameters Reference

Complete reference for all ComBat parameters.

## Constructor Parameters

### `batch` (required)

**Type**: `array-like`, `pd.Series`

Vector indicating batch assignment for each sample.

```python
batch = pd.Series(["A", "A", "B", "B", "C"], name="batch")
combat = ComBat(batch=batch)
```

**Requirements**:
- Must have same length as X
- At least 2 samples per batch
- Can be any hashable type (str, int, etc.)

### `method`

**Type**: `str`, default=`"johnson"`

**Options**: `"johnson"`, `"fortin"`, `"chen"`

Selects the ComBat algorithm variant. See [Methods](methods.md) for details.

```python
combat = ComBat(batch=batch, method="fortin")
```

### `parametric`

**Type**: `bool`, default=`True`

Use parametric empirical Bayes (faster, assumes normality).

```python
# Parametric (default)
combat = ComBat(batch=batch, parametric=True)

# Non-parametric (slower, no assumptions)
combat = ComBat(batch=batch, parametric=False)
```

### `mean_only`

**Type**: `bool`, default=`False`

If True, only correct batch means (preserve variance).

```python
combat = ComBat(batch=batch, mean_only=True)
```

**Use cases**:
- Preserve variance structure
- Less aggressive correction
- When variance differences are biological

### `reference_batch`

**Type**: `str` or `None`, default=`None`

Reference batch to match other batches to.

```python
combat = ComBat(batch=batch, reference_batch="Batch_A")
```

**Behavior**:
- Samples in reference batch unchanged
- Other batches adjusted to match reference
- Reference batch must exist in data

### `eps`

**Type**: `float`, default=`1e-8`

Small value added to variances to prevent division by zero.

```python
combat = ComBat(batch=batch, eps=1e-8)
```

Usually no need to change this parameter.

## Covariate Parameters

### `discrete_covariates`

**Type**: `array-like`, `pd.Series`, `pd.DataFrame`, or `None`, default=`None`

Categorical covariates to preserve (e.g., sex, diagnosis).

```python
sex = pd.DataFrame({"sex": ["M", "F", "M", "F"]})
diagnosis = pd.DataFrame({"dx": ["healthy", "disease", "healthy", "disease"]})

combat = ComBat(
    batch=batch,
    method="fortin",
    discrete_covariates=pd.concat([sex, diagnosis], axis=1)
)
```

**Supported by**: `"fortin"` and `"chen"` methods only

### `continuous_covariates`

**Type**: `array-like`, `pd.Series`, `pd.DataFrame`, or `None`, default=`None`

Continuous covariates to preserve (e.g., age, BMI).

```python
age = pd.DataFrame({"age": [25, 30, 45, 50]})
bmi = pd.DataFrame({"bmi": [22.5, 25.1, 28.3, 24.7]})

combat = ComBat(
    batch=batch,
    method="fortin",
    continuous_covariates=pd.concat([age, bmi], axis=1)
)
```

**Supported by**: `"fortin"` and `"chen"` methods only

## Chen Method Parameters

### `covbat_cov_thresh`

**Type**: `float` or `int`, default=`0.9`

Controls PCA dimensionality for Chen method.

**Float (0, 1]**: Cumulative variance threshold
```python
combat = ComBat(
    batch=batch,
    method="chen",
    covbat_cov_thresh=0.95  # Retain 95% variance
)
```

**Int >= 1**: Exact number of components
```python
combat = ComBat(
    batch=batch,
    method="chen",
    covbat_cov_thresh=50  # Use 50 components
)
```

**Only used by**: `"chen"` method

## Visualization Parameters

### `plot_transformation()`

Visualize the batch correction effect.

#### Parameters

**`X`** (required): Input data to visualize

**`reduction_method`**: `'pca'`, `'tsne'`, or `'umap'`, default=`'pca'`

Dimensionality reduction method for visualization.

**`n_components`**: `2` or `3`, default=`2`

Number of dimensions to visualize.

**`plot_type`**: `'static'` or `'interactive'`, default=`'static'`

- `'static'`: matplotlib plots
- `'interactive'`: plotly plots

**`figsize`**: `tuple`, default=`(12, 5)`

Figure size for static plots.

**`alpha`**: `float`, default=`0.7`

Point transparency (0-1).

**`point_size`**: `int`, default=`50`

Size of scatter plot points.

**`cmap`**: `str`, default=`'Set1'`

Matplotlib colormap name.

**`title`**: `str` or `None`, default=`None`

Plot title.

**`show_legend`**: `bool`, default=`True`

Show legend.

**`return_embeddings`**: `bool`, default=`False`

If True, return embeddings along with figure.

**`**reduction_kwargs`**: Additional parameters for reduction methods.

#### Example

```python
fig = combat.plot_transformation(
    X,
    reduction_method='tsne',
    n_components=2,
    plot_type='interactive',
    perplexity=40,  # t-SNE parameter
    return_embeddings=False
)
```

## Parameter Validation

combatlearn validates parameters and raises informative errors:

```python
# Invalid method
ComBat(batch=batch, method="invalid")
# ValueError: method must be 'johnson', 'fortin', or 'chen'

# Invalid covbat_cov_thresh (float)
ComBat(batch=batch, method="chen", covbat_cov_thresh=1.5)
# ValueError: covbat_cov_thresh must be in (0, 1] when float

# Invalid covbat_cov_thresh (int)
ComBat(batch=batch, method="chen", covbat_cov_thresh=0)
# ValueError: covbat_cov_thresh must be >= 1 when int

# Reference batch doesn't exist
ComBat(batch=batch, reference_batch="NonExistent")
# ValueError: reference_batch='NonExistent' not present in batches
```

## Default Values Summary

| Parameter | Default | Type |
|-----------|---------|------|
| `method` | `"johnson"` | str |
| `parametric` | `True` | bool |
| `mean_only` | `False` | bool |
| `reference_batch` | `None` | str or None |
| `eps` | `1e-8` | float |
| `discrete_covariates` | `None` | DataFrame or None |
| `continuous_covariates` | `None` | DataFrame or None |
| `covbat_cov_thresh` | `0.9` | float or int |

## Next Steps

- See [Methods](methods.md) for algorithm details
- Learn about [Cross-Validation](cross-validation.md)
- Try [Examples](../examples/basic-usage.md)

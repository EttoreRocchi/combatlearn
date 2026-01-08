# Visualization Guide

combatlearn provides built-in tools to visualize batch effects before and after correction.

## Quick Start

```python
from combatlearn import ComBat
import matplotlib.pyplot as plt

# Fit ComBat
combat = ComBat(batch=batch, method="johnson").fit(X)

# Visualize
fig = combat.plot_transformation(X, reduction_method='pca')
plt.show()
```

## Dimensionality Reduction Methods

### PCA (Principal Component Analysis)

Fast, linear method. Best for initial exploration.

```python
fig = combat.plot_transformation(
    X,
    reduction_method='pca',
    n_components=2
)
```

### t-SNE

Non-linear method. Good for revealing cluster structure.

```python
fig = combat.plot_transformation(
    X,
    reduction_method='tsne',
    n_components=2,
    perplexity=30,  # t-SNE parameter
    max_iter=1000
)
```

### UMAP

Modern non-linear method. Preserves global and local structure.

```python
fig = combat.plot_transformation(
    X,
    reduction_method='umap',
    n_components=2,
    n_neighbors=15,  # UMAP parameter
    min_dist=0.1
)
```

## 2D vs 3D Visualization

### 2D Plots (default)

```python
fig = combat.plot_transformation(X, n_components=2)
```

### 3D Plots

```python
fig = combat.plot_transformation(X, n_components=3)
```

## Static vs Interactive

### Static (Matplotlib)

```python
fig = combat.plot_transformation(
    X,
    plot_type='static',
    figsize=(14, 6),
    alpha=0.7,
    point_size=60
)
plt.savefig('combat_effect.png', dpi=300)
```

### Interactive (Plotly)

```python
fig = combat.plot_transformation(
    X,
    plot_type='interactive'
)
fig.write_html('combat_effect.html')
```

## Customization

### Colors

```python
fig = combat.plot_transformation(
    X,
    cmap='Set2',  # matplotlib colormap
    alpha=0.8
)
```

### Titles and Labels

```python
fig = combat.plot_transformation(
    X,
    title='Batch Effect Correction Analysis',
    show_legend=True
)
```

## Retrieving Embeddings

Get the reduced-dimension coordinates for custom plotting:

```python
fig, embeddings = combat.plot_transformation(
    X,
    return_embeddings=True
)

# embeddings['original'] - before correction
# embeddings['transformed'] - after correction

print(embeddings['original'].shape)  # (n_samples, n_components)
```

## Custom Plotting Example

```python
import matplotlib.pyplot as plt

# Get embeddings
fig, embeddings = combat.plot_transformation(
    X,
    reduction_method='pca',
    return_embeddings=True
)

# Custom plot
fig, ax = plt.subplots(figsize=(10, 6))

for batch_name in batch.unique():
    mask = batch == batch_name
    ax.scatter(
        embeddings['transformed'][mask, 0],
        embeddings['transformed'][mask, 1],
        label=batch_name,
        alpha=0.6,
        s=100,
        edgecolors='black'
    )

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Custom ComBat Visualization')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Complete Workflow

```python
from combatlearn import ComBat
import matplotlib.pyplot as plt

# Fit ComBat
combat = ComBat(batch=batch, method="fortin",
                continuous_covariates=age).fit(X)

# Create multiple visualizations
methods = ['pca', 'tsne', 'umap']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, method in enumerate(methods):
    combat_fig = combat.plot_transformation(
        X,
        reduction_method=method,
        plot_type='static'
    )
    plt.sca(axes[i])

plt.tight_layout()
plt.savefig('batch_effects_comparison.png', dpi=300)
```

## Next Steps

- See [Metrics Guide](metrics.md) for quantitative assessment
- See complete [Examples](../examples/basic-usage.md)
- Check [API Reference](../api.md) for all parameters

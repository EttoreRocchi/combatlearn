# Batch Effect Metrics Guide

combatlearn provides comprehensive metrics to quantitatively assess batch correction quality. This guide explains each metric, its interpretation, and how to use them effectively.

## Quick Start

```python
from combatlearn import ComBat

# Option 1: Auto-compute during fit_transform
combat = ComBat(batch=batch, method="johnson", compute_metrics=True)
X_corrected = combat.fit_transform(X)

# Access cached metrics
metrics = combat.metrics_

# Option 2: Compute on-demand for any data
combat = ComBat(batch=batch, method="johnson")
combat.fit(X_train)
metrics = combat.compute_batch_metrics(X_test, batch=test_batch)
```

## Metrics Overview

The metrics are organized into three categories:

| Category | Purpose | Goal After Correction |
|----------|---------|----------------------|
| **Batch Effect** | Measure batch separation | Lower separation |
| **Preservation** | Measure structure retention | Higher retention |
| **Alignment** | Measure batch alignment | Better alignment |

## Batch Effect Quantification

These metrics measure how well batches are mixed after correction. The goal is to reduce batch-driven clustering.

### Silhouette Coefficient

**Range:** [-1, 1]
**Ideal after correction:** Near 0

The silhouette coefficient measures how similar samples are to their own batch compared to other batches. It's computed using batch labels as cluster assignments.

- **+1**: Samples are well-separated by batch (strong batch effect)
- **0**: Samples are evenly distributed across batches (good mixing)
- **-1**: Samples are closer to other batches (over-correction)

```python
silhouette = metrics['batch_effect']['silhouette']
print(f"Before: {silhouette['before']:.3f}")  # High = batch effect
print(f"After: {silhouette['after']:.3f}")    # Should be ~0
```

### Davies-Bouldin Index

**Range:** [0, $\infty$)
**Ideal after correction:** High values

The Davies-Bouldin index measures the ratio of within-cluster scatter to between-cluster separation, using batch labels as clusters.

- **Low values**: Well-defined, separated batch clusters (strong batch effect)
- **High values**: Poorly-defined batch clusters, more overlap (good - batch effect removed)

This metric is sensitive to cluster shape and size, making it complementary to silhouette.

```python
db = metrics['batch_effect']['davies_bouldin']
print(f"Before: {db['before']:.3f}, After: {db['after']:.3f}")
# Higher is better - batches should not form distinct clusters
```

### kBET (k-nearest neighbor Batch Effect Test)

**Range:** [0, 1]
**Ideal after correction:** Near 1

kBET tests whether the batch composition in local neighborhoods matches the global batch proportions. For each sample, it examines the k-nearest neighbors and performs a chi-squared test.

- **1.0**: All neighborhoods have expected batch proportions (perfect mixing)
- **0.0**: No neighborhoods have expected batch proportions (complete separation)

**Reference:** Büttner et al. 2019, "A test metric for assessing single-cell RNA-seq batch correction", Nature Methods.

```python
kbet = metrics['batch_effect']['kbet']
print(f"Acceptance rate: {kbet['before']:.1%} → {kbet['after']:.1%}")
```

**Parameters:**
```python
metrics = combat.compute_batch_metrics(
    X,
    kbet_k0=50  # Number of neighbors (default: 10% of samples)
)
```

### LISI (Local Inverse Simpson's Index)

**Range:** [1, n_batches]
**Ideal after correction:** Near n_batches

LISI measures the effective number of batches represented in each sample's neighborhood. It uses a probability distribution over batches weighted by distance.

- **1**: Only one batch in neighborhood (poor mixing)
- **n_batches**: All batches equally represented (perfect mixing)

**Reference:** Korsunsky et al. 2019, "Fast, sensitive and accurate integration of single-cell data with Harmony", Nature Methods.

```python
lisi = metrics['batch_effect']['lisi']
n_batches = lisi['max_value']
print(f"LISI: {lisi['before']:.2f} → {lisi['after']:.2f} (max: {n_batches})")
```

**Parameters:**
```python
metrics = combat.compute_batch_metrics(
    X,
    lisi_perplexity=30  # Controls neighborhood size
)
```

### Variance Ratio

**Range:** [0, $\infty$)
**Ideal after correction:** Near 0

An ANOVA-style metric measuring the ratio of between-batch variance to within-batch variance. This directly quantifies how much of the total variance is explained by batch.

- **0**: No variance explained by batch
- **Higher**: More variance explained by batch (stronger batch effect)

```python
vr = metrics['batch_effect']['variance_ratio']
print(f"Before: {vr['before']:.3f}, After: {vr['after']:.3f}")
```

## Structure Preservation

These metrics ensure that biological relationships between samples are maintained after correction.

### k-NN Preservation

**Range:** [0, 1]
**Ideal:** Near 1

Measures what fraction of each sample's k-nearest neighbors are preserved after correction. This is computed for multiple k values.

- **1.0**: All neighbors preserved (perfect structure preservation)
- **0.0**: No neighbors preserved (complete structure loss)

```python
knn = metrics['preservation']['knn']
for k, preservation in knn.items():
    print(f"k={k}: {preservation:.1%} neighbors preserved")
```

**Parameters:**
```python
metrics = combat.compute_batch_metrics(
    X,
    k_neighbors=[5, 10, 25, 50]  # Multiple k values
)
```

Smaller k values (5-10) capture local structure, while larger values (50+) capture global relationships.

### Distance Correlation

**Range:** [-1, 1]
**Ideal:** Near 1

Spearman correlation between pairwise distances before and after correction. This measures whether the relative distances between all pairs of samples are preserved.

- **1.0**: Perfect rank preservation of distances
- **0.0**: No correlation (random reordering)
- **-1.0**: Complete reversal (unlikely)

```python
dc = metrics['preservation']['distance_correlation']
print(f"Distance correlation: {dc:.3f}")
```

For large datasets, distances are computed on a subsample (default: 1000 samples) for efficiency.

## Alignment Metrics

These metrics assess how well batch centers and variances are aligned after correction.

### Centroid Distance

**Direction:** Lower is better

Mean pairwise Euclidean distance between batch centroids (means). Lower values indicate batch centers are more aligned.

```python
cd = metrics['alignment']['centroid_distance']
print(f"Centroid separation: {cd['before']:.2f} → {cd['after']:.2f}")
```

### Levene Statistic

**Direction:** Lower is better

Median of Levene's test statistics across all features. Levene's test assesses equality of variances across batches.

- **Lower**: More homogeneous variance across batches
- **Higher**: Heterogeneous variance (batch-specific variance)

```python
lev = metrics['alignment']['levene_statistic']
print(f"Levene statistic: {lev['before']:.2f} → {lev['after']:.2f}")
```

## Complete Example

```python
from combatlearn import ComBat
import pandas as pd

# Load data
X = pd.read_csv('expression_data.csv', index_col=0)
batch = pd.read_csv('batch_labels.csv', index_col=0)['batch']

# Fit ComBat with metrics
combat = ComBat(
    batch=batch,
    method='fortin',
    continuous_covariates=age,
    compute_metrics=True
)
X_corrected = combat.fit_transform(X)

# Display results
metrics = combat.metrics_

print("=" * 50)
print("BATCH EFFECT CORRECTION REPORT")
print("=" * 50)

print("\n--- Batch Effect Metrics ---")
for name, vals in metrics['batch_effect'].items():
    if name == 'lisi':
        print(f"{name:20s}: {vals['before']:8.3f} → {vals['after']:8.3f} (max: {vals['max_value']})")
    else:
        print(f"{name:20s}: {vals['before']:8.3f} → {vals['after']:8.3f}")

print("\n--- Structure Preservation ---")
print(f"Distance correlation: {metrics['preservation']['distance_correlation']:.3f}")
print("k-NN preservation:")
for k, v in metrics['preservation']['knn'].items():
    print(f"  k={k:3d}: {v:.1%}")

print("\n--- Alignment ---")
for name, vals in metrics['alignment'].items():
    print(f"{name:20s}: {vals['before']:8.3f} → {vals['after']:8.3f}")
```

## Metrics on New Data

You can compute metrics on new data with different batch labels:

```python
# Fit on training data
combat = ComBat(batch=train_batch, method='johnson')
combat.fit(X_train)

# Evaluate on test data
test_metrics = combat.compute_batch_metrics(
    X_test,
    batch=test_batch,  # Different batch labels
    k_neighbors=[5, 10, 25]
)
```

## PCA Preprocessing

By default, metrics are computed in the original feature space. For high-dimensional data, you can optionally reduce dimensionality using PCA before computing metrics:

```python
# Default: compute in original feature space
metrics = combat.compute_batch_metrics(X)

# Optional: reduce to PCA space for computational efficiency
metrics = combat.compute_batch_metrics(
    X,
    pca_components=50  # Must be < min(n_samples, n_features)
)
```

Using PCA can help with computational efficiency and noise reduction for high-dimensional datasets.

## Next Steps

- See [Visualization Guide](visualization.md) for visual assessment
- Check [API Reference](../api.md) for all parameters
- Review [Examples](../examples/basic-usage.md) for practical workflows

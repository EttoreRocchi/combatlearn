# ComBat Methods

combatlearn implements three variants of the ComBat algorithm. This guide helps you choose the right method for your use case.

## Johnson Method (Classic ComBat)

**Reference**: Johnson et al. (2007)

The original ComBat algorithm without covariate support.

### When to Use

- Simple batch correction scenarios
- No biological covariates to preserve
- Exploratory data analysis
- Fastest computation time

### Algorithm

1. Standardize features across all samples
2. Estimate location (γ) and scale (δ) parameters for each batch
3. Apply empirical Bayes shrinkage
4. Remove batch effects using adjusted parameters

### Example

```python
from combatlearn import ComBat

combat = ComBat(
    batch=batch,
    method="johnson",
    parametric=True  # or False for non-parametric
)
X_corrected = combat.fit_transform(X)
```

### Advantages

✅ Simple and fast
✅ No covariate dependencies
✅ Well-established method

### Limitations

❌ Cannot preserve covariate effects

## Fortin Method (neuroCombat)

**Reference**: Fortin et al. (2018)

Extended ComBat that preserves effects of biological covariates.

### When to Use

- Known biological variables (age, sex, diagnosis)
- Need to preserve biological variation
- Recommended for most applications
- Standard choice for neuroimaging

### Algorithm

1. Build design matrix with batch indicators and covariates
2. Estimate batch effects while accounting for covariates
3. Apply empirical Bayes shrinkage
4. Remove only batch-related variation

### Example

```python
from combatlearn import ComBat
import pandas as pd

# Define covariates
age = pd.DataFrame({"age": [25, 30, 45, ...]})
sex = pd.DataFrame({"sex": ["M", "F", "M", ...]})
diagnosis = pd.DataFrame({"dx": ["healthy", "disease", ...]})

combat = ComBat(
    batch=batch,
    method="fortin",
    continuous_covariates=age,
    discrete_covariates=pd.concat([sex, diagnosis], axis=1)
)
X_corrected = combat.fit_transform(X)
```

### Advantages

✅ Preserves covariate effects
✅ Removes only technical variation
✅ More biologically meaningful

### Limitations

❌ Requires covariate information
❌ Slightly slower than Johnson

## Chen Method (CovBat)

**Reference**: Chen et al. (2022)

PCA-based ComBat that operates in reduced dimensionality space.

### When to Use

- High-dimensional data (many features)
- Batch effects vary across features
- Feature-specific corrections needed
- Computational efficiency important

### Algorithm

1. Apply Fortin method for mean/variance adjustment
2. Perform PCA on corrected data
3. Apply batch correction in PC space
4. Transform back to original space

### Example

```python
from combatlearn import ComBat

combat = ComBat(
    batch=batch,
    method="chen",
    continuous_covariates=age,
    discrete_covariates=sex,
    covbat_cov_thresh=0.95  # Retain 95% variance
)
X_corrected = combat.fit_transform(X)
```

### Variance Threshold Options

You can specify the number of principal components in two ways:

**Option 1: Cumulative Variance (float)**
```python
covbat_cov_thresh=0.95  # Retain 95% of variance
```

**Option 2: Fixed Number (int)**
```python
covbat_cov_thresh=50  # Use exactly 50 components
```

### Advantages

✅ Handles high-dimensional data
✅ Feature-specific corrections
✅ Can reduce dimensionality
✅ Preserves covariate effects

### Limitations

❌ Requires covariate information
❌ Most computationally intensive
❌ Information loss in PCA step

## Parametric vs Non-Parametric

All methods support both parametric and non-parametric empirical Bayes:

**Parametric** (default):
- Faster computation
- Assumes normal distribution
- Recommended for most datasets

**Non-Parametric**:
- Iterative scheme
- No distribution assumptions
- Use when parametric assumptions violated

```python
# Parametric (default)
combat = ComBat(batch=batch, method="fortin", parametric=True)

# Non-parametric
combat = ComBat(batch=batch, method="fortin", parametric=False)
```

## Mean-Only Correction

All methods support mean-only mode, which corrects batch means but preserves variance:

```python
combat = ComBat(
    batch=batch,
    method="fortin",
    mean_only=True  # Only correct means
)
```

**Use when**: You want to preserve variance structure across batches.

## Reference Batch

Optionally specify a reference batch. Other batches will be adjusted to match it:

```python
combat = ComBat(
    batch=batch,
    method="johnson",
    reference_batch="Batch_A"  # Match to Batch_A
)
```

Samples in the reference batch remain unchanged after correction.

## Choosing a Method

**Simple Decision Tree**:

1. **No covariates?** → Use Johnson
2. **Have covariates + low/normal dimensionality?** → Use Fortin
3. **Have covariates + high dimensionality?** → Use Chen

## Next Steps

- See the [API Reference](api.rst) for complete parameter documentation

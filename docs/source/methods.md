# Method Guide

combatlearn implements four variants of the ComBat algorithm. This guide helps you choose the right method for your use case.

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

## Longitudinal Method (Longitudinal ComBat)

**Reference**: Beer et al. (2020)

ComBat for repeated-measures / longitudinal designs, where the same subjects are measured more than once (e.g. across visits or time points). It extends the Fortin model with a per-subject **random intercept** - a subject-specific offset shared across that subject's repeated measurements - so within-subject correlation is accounted for when estimating batch effects rather than being treated as independent noise.

As in the other methods, `batch` is still the technical batch you want to remove (e.g. scanner or site); what makes a design *longitudinal* is that the same subjects recur, identified by `subject_id`.

### When to Use

- Harmonising a complete repeated-measures cohort in one pass (`fit_transform`) for downstream analysis - the primary use case
- Subjects measured at multiple time points, or across multiple batches
- Repeated-measures / panel / multi-visit designs
- You want within-subject correlation modeled rather than ignored

### Algorithm

1. Fit the fixed-effects mean model (grand mean + covariates + batch) with a per-subject random intercept, estimated by REML
2. Standardize using the covariate fit plus each subject's estimated random intercept (BLUP)
3. Estimate location (γ) and scale (δ) parameters per batch and apply empirical Bayes shrinkage
4. Remove batch effects, leaving subject and covariate structure intact

### Example

```python
from combatlearn import ComBat

combat = ComBat(
    batch=batch,
    method="longitudinal",
    subject_id=subject,         # required: one label per sample
    time_covariate=time,        # optional: continuous time variable
    continuous_covariates=age,  # optional
    discrete_covariates=sex,    # optional
)
X_corrected = combat.fit_transform(X)
```

### Advantages

✅ Accounts for within-subject correlation (repeated measures)
✅ Preserves covariate effects (like Fortin)
✅ Same scikit-learn `fit`/`transform` contract; supports parametric and non-parametric EB, `mean_only`, and `reference_batch`

### Limitations

❌ Requires a `subject_id` for every sample
❌ Subjects unseen at fit are corrected with a zero random intercept (i.e. the population mean)

### In-sample harmonization vs. new-subject prediction

Longitudinal ComBat's advantage - separating within- from between-subject variance and modelling each subject's own intercept - is realised **in-sample**, while a subject is present at fit time. Its intended use is **transductive harmonization of a complete repeated-measures cohort**: call `fit_transform(X)` on the whole dataset so every subject gets its random intercept, then run your downstream (typically group-level) analysis on the harmonised data.

For a subject **unseen at fit** - which is every test subject under correct subject-grouped cross-validation (`GroupKFold` on `subject_id`) - the random intercept is zero by construction, so the correction reduces toward Fortin, but with scale parameters calibrated on subject-centred residuals applied to data that still carries the between-subject spread. In that setting it offers little over Fortin and can even harmonise slightly worse. **For cross-validated prediction that must generalise to new subjects, prefer `method="fortin"`; reserve `method="longitudinal"` for harmonising a full repeated-measures dataset for downstream analysis.**

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

1. **Harmonising a complete repeated-measures cohort in-sample?** → Use Longitudinal (for cross-validated prediction on *new* subjects, use Fortin)
2. **No covariates?** → Use Johnson
3. **Have covariates + low/normal dimensionality?** → Use Fortin
4. **Have covariates + high dimensionality?** → Use Chen

## Next Steps

- See the [API Reference](api) for complete parameter documentation

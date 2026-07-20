# Method Guide

combatlearn implements several variants of the ComBat algorithm. This guide helps you choose the right method for your use case.

Each `method` also accepts case- and separator-insensitive literature aliases: `classic_combat` (johnson), `neurocombat` (fortin), `covbat` (chen), `longcombat` (longitudinal), `combat_gam` (gam), `covbatgam` (covbat_gam).

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

```{note}
`ComBat(method="longitudinal")` is **deprecated** since v2.3.0 (it emits a `DeprecationWarning`) and will be removed from the inductive `ComBat` in v3.0.0. Longitudinal ComBat is a whole-cohort harmonizer whose benefit is in-sample, so its home is now `combatlearn.transductive.TransductiveComBat(method="longitudinal")`. The algorithm below is unchanged; only the entry point moves. See [Transductive ComBat](#transductive-combat-whole-cohort-harmonization).
```

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

## GAM Methods (ComBat-GAM and CovBat-GAM)

**Reference**: Pomponio et al. (2020)

`method="gam"` is Fortin with the continuous covariates modeled **nonlinearly** using B-spline bases, and `method="covbat_gam"` is the same nonlinear covariate model inside CovBat (the Chen PCA step). This matters when a biological covariate has a nonlinear effect on the features - the canonical case is **age across the lifespan**, which is markedly nonlinear.

Unlike Longitudinal ComBat, the GAM methods are fully **inductive and cross-validation-safe**: the spline knots are learned from the training data and stored, the basis is rebuilt from those knots at transform time, and held-out values outside the training range are clamped to the boundary knots. They use the same `fit`/`transform` contract as Fortin/Chen.

### When to Use

- A continuous covariate (e.g. age) has a nonlinear relationship with the features
- The covariate is correlated with batch (e.g. sites recruited different age ranges), where a linear model would confound the nonlinear covariate effect with the batch effect
- Use `gam` for the Fortin setting and `covbat_gam` for the high-dimensional CovBat setting

### Algorithm

1. Replace each continuous covariate listed in `smooth_terms` with its B-spline basis (degree-`spline_degree`, `spline_df` basis functions; interior knots at training quantiles, boundary knots at the data range or `smooth_term_bounds`). One basis column is dropped per term for identifiability against the batch indicators.
2. Run the unchanged Fortin (`gam`) or CovBat (`covbat_gam`) empirical-Bayes pipeline on the resulting design.

### Example

```python
from combatlearn import ComBat
import pandas as pd

age = pd.DataFrame({"age": [25, 30, 45, ...]})  # nonlinear lifespan effect

combat = ComBat(
    batch=batch,
    method="gam",
    continuous_covariates=age,
    discrete_covariates=sex,
    smooth_terms=["age"],   # default None smooths all continuous covariates
    spline_df=10,           # B-spline degrees of freedom (default 10)
    spline_degree=3,        # cubic (default 3)
    # smooth_term_bounds=(0, 100),  # optional; widen to cover held-out data
)
X_corrected = combat.fit_transform(X)
```

### Advantages

✅ Captures nonlinear covariate effects (e.g. age over the lifespan)
✅ Avoids confounding a nonlinear covariate with batch when the two are correlated
✅ Inductive and cross-validation-safe, like Fortin/Chen
✅ No new runtime dependency (B-splines built with SciPy)

### Limitations

❌ Requires at least one continuous covariate (raises otherwise)
❌ A smooth term needs enough distinct values to support `spline_df` (raises otherwise)
❌ Slightly more parameters to estimate than the linear Fortin/Chen models

## NestedComBat (Nested / OPNested / GMM ComBat)

**Reference**: Horng et al. (2022)

The methods above take a single batch variable. When several technical variables need harmonizing at once - for example **site**, **scanner**, and **protocol** - concatenating them into one batch variable creates many tiny groups and unstable estimates. `NestedComBat` instead applies single-batch ComBat to each variable **in sequence**, feeding the output of one step into the next. It is a separate scikit-learn transformer (not a `method=` option), and it is inductive and cross-validation-safe like `ComBat`: the harmonization order, the optional Gaussian-mixture grouping, and every per-step parameter are learned on the training data and frozen for transform.

### When to Use

- More than one batch variable to remove (site + scanner + protocol, ...)
- You want the composition to preserve covariates across every step
- You suspect a latent (unlabeled) grouping in the data that a Gaussian mixture can recover

### Algorithm

1. (Optional, `optimize_order=True`) Try harmonization orders and keep the one that leaves the fewest features with a significant residual batch effect (Anderson-Darling k-sample test, summed over all batch variables). The search is exhaustive over all `k!` orderings up to `max_exhaustive_vars` (default 4); above the cap it warns and falls back to greedy forward selection.
2. Apply single-batch ComBat to each batch variable in the chosen order, each step delegating to a `ComBatModel` with the configured `method` (`"fortin"`/`"chen"`/`"gam"`/`"covbat_gam"`), preserving the discrete/continuous covariates.
3. (Optional, `gmm`) Fit a 2-component Gaussian mixture per feature (unbalanced splits filtered by `gmm_min_cluster_frac`, best feature by AIC) and feed its latent grouping in as an extra batch variable (`gmm="batch"`, `+GMM`) or a protected covariate (`gmm="covariate"`, `-GMM`).

### Example

```python
import pandas as pd
from combatlearn import NestedComBat

# One column per batch variable
batch = pd.DataFrame({"site": site, "scanner": scanner, "protocol": protocol})

nested = NestedComBat(
    batch=batch,
    continuous_covariates=age,
    discrete_covariates=sex,
    method="fortin",       # per-step engine
    optimize_order=True,   # OPNested order optimization
    max_exhaustive_vars=4, # exhaustive up to 4 variables, else greedy
    gmm=None,              # or "batch" (+GMM) / "covariate" (-GMM)
)
X_corrected = nested.fit_transform(X)
print(nested.order_)  # the chosen harmonization order
```

With `reference_batch`, pass a `{batch_variable: level}` dict so each step keeps its own reference level unchanged (a bare string is accepted only when there is a single batch variable).

### Advantages

✅ Harmonizes several batch variables without exploding them into tiny groups
✅ Composes with the Fortin, CovBat, and GAM engines and preserves covariates across steps
✅ Order optimization targets the residual batch effect directly
✅ Inductive and cross-validation-safe, like `ComBat`

### Limitations

❌ Only the covariate-aware engines are allowed (`johnson`/`longitudinal` are not)
❌ The exhaustive order search is factorial; large numbers of batch variables fall back to greedy
❌ The Gaussian-mixture grouping needs a balanced two-cluster split to be added

## Transductive ComBat (Whole-Cohort Harmonization)

Some ComBat variants only pay off **in-sample**: their benefit is tied to the samples present at fit time, so there is no leakage-free way to freeze them on a training split and apply them to held-out data. These live in `combatlearn.transductive.TransductiveComBat`, a single class with a `method=` selector (mirroring `ComBat`), used as a one-shot `fit_transform` over a complete cohort. Calling `transform` on separate held-out data raises, and it is deliberately not a `Pipeline` step.

Currently `method="longitudinal"` (Longitudinal ComBat, Beer et al. 2020) is available; the algorithm is exactly the one described under [Longitudinal Method](#longitudinal-method-longitudinal-combat).

```{note}
"Transductive" here follows scikit-learn's glossary sense: a method that "is designed to model a specific dataset, but not to apply that model to unseen data" - i.e. whole-cohort, non-inductive, `fit_transform`-only. It is *not* the strictly supervised Vapnik sense of transduction (predicting labels for specific unlabeled points): ComBat is unsupervised and predicts no labels.
```

### Example

```python
from combatlearn.transductive import TransductiveComBat

harmonizer = TransductiveComBat(
    batch=batch,
    method="longitudinal",
    subject_id=subject,         # required: one label per sample
    time_covariate=time,        # optional
    continuous_covariates=age,  # optional
    discrete_covariates=sex,    # optional
)
X_corrected = harmonizer.fit_transform(X)  # whole cohort, one pass
```

### When to Use

- Harmonizing a **complete** repeated-measures cohort in one pass for downstream (typically group-level) analysis
- You do not need to apply the frozen correction to new, unseen samples

For cross-validated prediction that must generalize to **new subjects**, use the inductive `ComBat` (e.g. `method="fortin"`) instead.

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

1. **Several batch variables to remove (site + scanner + ...)?** → Use `NestedComBat` (with any of the engines below as its per-step `method`)
2. **Harmonising a complete repeated-measures cohort in-sample?** → Use `TransductiveComBat(method="longitudinal")` (for cross-validated prediction on *new* subjects, use Fortin)
3. **No covariates?** → Use Johnson
4. **A continuous covariate has a nonlinear effect (e.g. age)?** → Use GAM (`gam`, or `covbat_gam` for high dimensionality)
5. **Have covariates + low/normal dimensionality?** → Use Fortin
6. **Have covariates + high dimensionality?** → Use Chen

## Next Steps

- See the [API Reference](api) for complete parameter documentation

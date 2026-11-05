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

Extends Fortin with a covariance correction: it harmonizes the principal-component scores of the Fortin residuals, bringing each batch's covariance toward the pooled one. The output keeps every feature.

### When to Use

- Batch effects also change the covariance (correlation) between features, not only their means and variances
- Many correlated features, where site differences show up in the joint distribution
- Machine-learning or multivariate analyses, which can still detect the batch from the covariance after Fortin
- The same covariate protection as Fortin is needed (covariates are optional)

### Algorithm

1. Apply Fortin method for mean/variance adjustment, then take the residuals with respect to the intercept and the covariate effects
2. Perform PCA on those residuals
3. Bring each batch's mean and variance of the retained PC scores to the pooled mean and variance
4. Add the change of the retained PC scores back to the residuals, then add the intercept and covariate effects back

Because the covariate effects are removed before the PCA (Chen et al., 2022, Steps 1-4), the covariance step cannot absorb a covariate effect that is unbalanced across batches. Only the retained PC scores change: the rest of each residual, including the part of a held-out sample outside the span of the training PCs, is kept as it is. The PCA and the per-PC parameters are learned at fit and frozen for transform.

Step 3 is a plain moment match of the scores, as described in the paper and as the reference R implementation does by default (`score.eb = FALSE`): no empirical-Bayes shrinkage is applied to them, because a prior pooled over the PCs of a batch would depend on the arbitrary sign of each PC. `parametric` still selects the empirical-Bayes variant of the Fortin step.

The PCA is on the covariance of the residuals, as in the paper (Step 2). The reference R implementation standardizes every feature first by default (`std.var = TRUE`). To give features on very different scales a comparable weight in the PCA, standardize them before harmonizing; this approximates `std.var = TRUE` but is not identical to it.

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
covbat_cov_thresh=50  # Use 50 components (at most those with training variance)
```

### Advantages

✅ Removes covariance batch effects that Fortin leaves in place
✅ Preserves covariate effects
✅ Keeps all features: PCs beyond `covbat_cov_thresh` pass through unchanged
✅ Inductive: the PCA and the per-PC parameters are frozen at fit

### Limitations

❌ Costlier than Fortin: a full PCA at fit
❌ Assumes the covariance batch effect lives in the variances of the PC scores (Chen et al., 2022); a batch effect on the correlation between PC scores is not targeted

## Longitudinal Method (Longitudinal ComBat)

**Reference**: Beer et al. (2020)

```{note}
Longitudinal ComBat has its own **inductive** transformer, `combatlearn.LongitudinalComBat`. Both older entry points are deprecated and will be removed in v3.0.0:

- `ComBat(method="longitudinal")` - deprecated since v2.3.0.
- `TransductiveComBat(method="longitudinal")` - deprecated since v2.4.0, because it is redundant: on a whole cohort every subject is seen at fit, so `LongitudinalComBat.fit_transform` returns bit-identical output *and* additionally supports `transform` on held-out data.

The algorithm below is unchanged in all three; only the entry point moves.
```

ComBat for repeated-measures / longitudinal designs, where the same subjects are measured more than once (e.g. across visits or time points). It extends the Fortin model with a per-subject **random intercept** - a subject-specific offset shared across that subject's repeated measurements - so within-subject correlation is accounted for when estimating batch effects rather than being treated as independent noise.

As in the other methods, `batch` is still the technical batch you want to remove (e.g. scanner or site); what makes a design *longitudinal* is that the same subjects recur, identified by `subject_id`.

### When to Use

- Repeated-measures / panel / multi-visit designs: subjects measured at multiple time points, or across multiple batches
- You want within-subject correlation modeled rather than ignored
- Harmonising a complete repeated-measures cohort in one pass (`fit_transform`) for downstream analysis
- **Incremental harmonization**: freezing the correction on a reference cohort and applying it to new visits of subjects already in that cohort, without re-correcting the existing data

The last case is where being inductive pays off; see [Preconditions and cross-validation](#preconditions-and-cross-validation) below for when it does not.

### Algorithm

1. Fit the fixed-effects mean model (grand mean + covariates + batch) with a per-subject random intercept, estimated by REML
2. Standardize using the covariate fit plus each subject's estimated random intercept (BLUP)
3. Estimate location (γ) and scale (δ) parameters per batch and apply empirical Bayes shrinkage
4. Remove batch effects, leaving subject and covariate structure intact

### Example

```python
from combatlearn import LongitudinalComBat

harmonizer = LongitudinalComBat(
    batch=batch,
    subject_id=subject,         # required: one label per sample
    time_covariate=time,        # optional: continuous time variable
    continuous_covariates=age,  # optional
    discrete_covariates=sex,    # optional
)

# Whole cohort in one pass ...
X_corrected = harmonizer.fit_transform(X)

# ... or fit once and apply the frozen correction to later visits
harmonizer.fit(X_reference)
X_new_corrected = harmonizer.transform(X_new_visits)
```

Use `on_unseen_subject="error"` to refuse samples whose subject was absent at fit, instead of falling back to the population correction:

```python
harmonizer = LongitudinalComBat(
    batch=batch,
    subject_id=subject,
    on_unseen_subject="error",  # default "population" = zero random intercept
)
```

```{deprecated} 2.3.0
The equivalent `ComBat(batch=batch, method="longitudinal", subject_id=subject)` form still works but emits a `DeprecationWarning`; it is removed in v3.0.0.
```

### Advantages

✅ Accounts for within-subject correlation (repeated measures)
✅ Preserves covariate effects (like Fortin)
✅ Inductive and leakage-free: every parameter, including each subject's random intercept, is frozen at fit
✅ Same scikit-learn `fit`/`transform` contract; supports parametric and non-parametric EB, `mean_only`, and `reference_batch`

### Limitations

❌ Requires a `subject_id` for every sample
❌ Its precondition is stricter than Fortin's (see below): full benefit needs the subject, not just the batch, to have been seen at fit
❌ Subjects unseen at fit degrade to a zero random intercept (the population correction), unless `on_unseen_subject="error"`

### Preconditions and cross-validation

Longitudinal ComBat is **inductive under a stricter precondition** than the other engines. `ComBat` requires the **batch** to have been seen at fit; `LongitudinalComBat` requires the **batch *and* the subject** to have been seen at fit. Outside that precondition it is not undefined - it degrades gracefully to the population correction (BLUP 0), which is exactly the empirical-Bayes posterior mean for a subject with no observations.

Harmonization and cross-validation then interact in three distinct ways, and only the first is an argument against this estimator:

| Regime | BLUP | Harmonizer leakage | Downstream eval bias | Advantage over Fortin |
|---|---|---|---|---|
| **New subject entirely** (subject-wise CV, e.g. `GroupKFold` on `subject_id` - the clinically valid split) | 0 | none | none | **none** |
| **Returning subject, new timepoint, no held-out set** (deployment / incremental harmonization) | available | none | n/a - there is no test set | **yes** |
| **Record-wise CV** (one subject's visits split across train and test) | available | none | **yes**, via subject-identity confounding | yes, but the benchmark is invalid |

Two different leakages have to be kept apart, and conflating them is what produced the older "longitudinal is transductive-only" advice:

- **Harmonizer leakage is nil in all three rows.** Subject *j*'s random intercept is estimated from *j*'s training visits only and frozen at fit, so applying it to *j*'s held-out visit consumes zero test information. The transform is leakage-free by construction.
- **Downstream evaluation bias is real in row 3** - but it is caused by the **split**, not by the harmonizer. Saeb et al. (2017, *GigaScience* 6(5), doi:[10.1093/gigascience/gix019](https://doi.org/10.1093/gigascience/gix019)) show that record-wise cross-validation substantially overestimates accuracy through subject-identity confounding, whether or not any harmonizer is in the pipeline. Fix the split; do not blame the transformer.

**Consequence.** The inductive value of Longitudinal ComBat is **incremental harmonization, not cross-validation** - row 2, which has peer-reviewed precedent as the motivating use case of iComBat (Tomo & Nakaki, 2025, *Computational and Structural Biotechnology Journal* 27:4121-4131, doi:[10.1016/j.csbj.2025.09.014](https://doi.org/10.1016/j.csbj.2025.09.014)): correcting newly added samples against a frozen reference cohort without re-correcting the old data. **For cross-validated prediction that must generalise to genuinely new subjects, prefer `ComBat(method="fortin")`.**

## GAM Methods (ComBat-GAM and CovBat-GAM)

**Reference**: Pomponio et al. (2020)

`method="gam"` is Fortin with the continuous covariates modeled **nonlinearly** using B-spline bases, and `method="covbat_gam"` is the same nonlinear covariate model inside CovBat (the Chen PCA step). This matters when a biological covariate has a nonlinear effect on the features - the canonical case is **age across the lifespan**, which is markedly nonlinear.

The GAM methods are **inductive and cross-validation-safe** under the same precondition as Fortin/Chen (batch seen at fit): the spline knots are learned from the training data and stored, the basis is rebuilt from those knots at transform time, and held-out values outside the training range are clamped to the boundary knots. They use the same `fit`/`transform` contract as Fortin/Chen.

### When to Use

- A continuous covariate (e.g. age) has a nonlinear relationship with the features
- The covariate is correlated with batch (e.g. sites recruited different age ranges), where a linear model would confound the nonlinear covariate effect with the batch effect
- Use `gam` for the Fortin setting and `covbat_gam` when batches also differ in feature covariance

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

## ComBat-seq (GLMComBat for Count Data)

**Reference**: Zhang, Parmigiani & Johnson (2020), *NAR Genomics and Bioinformatics* 2(3):lqaa078, [doi:10.1093/nargab/lqaa078](https://doi.org/10.1093/nargab/lqaa078)

Every method above assumes the data are Gaussian. RNA-seq counts are not: they are non-negative integers whose variance grows with the mean, and standardizing them destroys exactly the structure that count-based downstream tools expect. ComBat-seq replaces the Gaussian model with a **negative-binomial GLM** and, instead of shifting and rescaling residuals, maps each count through the CDF of its own batch's fitted distribution into the corresponding quantile of a batch-free distribution. Counts go in, counts come out.

`GLMComBat` is a **separate estimator** rather than another `method=` on `ComBat`, because the admissible domain of `X` changes (non-negative integers), `transform` returns integers, and the `parametric` empirical-Bayes switch has no analogue here.

It is **inductive and cross-validation-safe**. The quantile map is strictly per observation: the corrected value of a sample depends only on that sample's own counts, batch label, covariates and library size. Everything else - the per-feature background level, the batch effects, the covariate coefficients, the per-feature/per-batch dispersions and the batch-free target - is estimated once at `fit` and frozen.

### When to Use

- Raw RNA-seq (or other integer count) data, not normalized or log-transformed
- The downstream tool expects counts (DESeq2, edgeR, a count-aware classifier)
- You need harmonization inside a cross-validated pipeline, with `transform` applied to held-out samples

Use `ComBat` on log-CPM values instead if your downstream analysis is Gaussian anyway - the two are answering different questions.

### Algorithm

1. Drop features that are entirely zero within any batch; their batch means are zero and the quantile map is undefined. The mask is frozen at fit and those features pass through unchanged.
2. Estimate a per-feature, per-batch dispersion by maximizing the **Cox-Reid adjusted profile likelihood** (Cox & Reid 1987; McCarthy, Chen & Smyth 2012), starting from a common dispersion per batch.
3. Fit a negative-binomial log-linear model with a per-sample library-size offset, batch indicators and the covariate columns, to obtain the per-feature background level `alpha`, the batch effects `gamma`, and the covariate coefficients.
4. For every observation, map its count through the NB CDF of its own batch's fitted distribution and back through the inverse CDF of the batch-free distribution (batch-free mean, pooled dispersion). Counts of 0 and 1 are left alone, which preserves the zero-inflation structure.

### Example

```python
from combatlearn import GLMComBat

# counts: samples x genes, raw integer counts
harmonizer = GLMComBat(
    batch=batch,
    method="seq",
    discrete_covariates=tissue,      # ordinary covariates, preserved
    reference_batch="site_A",        # recommended under cross-validation
)

# Whole cohort in one pass - reproduces the reference implementation exactly
counts_corrected = harmonizer.fit_transform(counts)

# ... or fit once and apply the frozen correction to held-out samples
harmonizer.fit(counts_train)
counts_test_corrected = harmonizer.transform(counts_test)
```

### Advantages

✅ Models counts as counts - no Gaussian assumption, no log transform
✅ Integer in, integer out, so count-based downstream tools still apply
✅ Inductive and cross-validation-safe, unlike every published ComBat-seq implementation
✅ Reproduces InMoose `pycombat_seq` exactly on a whole cohort

### Limitations

❌ Requires raw counts; rejects negative or non-integer input
❌ Features all-zero in any batch are passed through uncorrected, and which features those are depends on the training fold (see [Cross-validation with the GLM engines](#cross-validation-with-the-glm-engines))
❌ Per-batch dispersions get noisy on small batches; a batch of one sample is rejected
❌ Slower than the Gaussian engines - the dispersion estimation is an iterative likelihood maximization per feature and per batch

## ComBat-met (GLMComBat for Methylation Data)

**Reference**: Wang (2025), *NAR Genomics and Bioinformatics* 7(2):lqaf062, [doi:10.1093/nargab/lqaf062](https://doi.org/10.1093/nargab/lqaf062)

DNA-methylation beta-values are proportions confined to `(0, 1)`, and a Gaussian correction can push them outside that interval, producing values that are not methylation fractions at all. ComBat-met is the same construction as ComBat-seq with the negative binomial replaced by a **beta regression** with a logit link: fit the mean and precision per feature and per batch, then map each value through the beta CDF into its batch-free quantile. There is no library-size term, since a proportion carries no depth.

### When to Use

- DNA-methylation beta-values (bounded 0-1) from arrays or aligned sequencing
- You need the corrected values to remain valid proportions
- You need `transform` on held-out samples inside a cross-validated pipeline

For M-values (the logit of beta-values, which are unbounded and roughly Gaussian) use `ComBat` directly - that is the standard alternative, and it is cheaper.

### Algorithm

1. Drop features the design already reproduces exactly - overall, or within any single batch. Such features have no residual variation for the regression to fit; the mask is frozen at fit and they pass through unchanged.
2. Estimate a per-feature, per-batch **precision** by maximizing the Cox-Reid adjusted profile likelihood, exactly as for the dispersion in ComBat-seq.
3. Fit a logit-link beta regression with batch indicators and the covariate columns, giving the per-feature batch effects on the mean and (through the precisions) on the spread.
4. Map every observation through the beta CDF of its own batch's fitted distribution and back through the inverse CDF of the batch-free distribution.

### Example

```python
from combatlearn import GLMComBat

# beta_values: samples x probes, each in [0, 1]
harmonizer = GLMComBat(
    batch=batch,
    method="met",
    discrete_covariates=sex,
    pseudo_beta=1e-4,   # substituted for exact 0 and 1, which the beta density excludes
)
beta_corrected = harmonizer.fit_transform(beta_values)
```

### Advantages

✅ Output stays strictly inside `(0, 1)` - always a valid methylation fraction
✅ Models the mean *and* the precision per batch, so it corrects spread as well as level
✅ Inductive and cross-validation-safe; the first implementation of ComBat-met in Python at all
✅ Reproduces the ComBat-met R package on committed fixtures

### Limitations

❌ Requires values in `[0, 1]`; exact 0 and 1 are replaced by `pseudo_beta`
❌ Constant or design-collinear features are passed through uncorrected
❌ Empirical-Bayes shrinkage is not offered: the method's author reports that it causes under-correction and recommends against it
❌ Slower than the Gaussian engines, for the same reason as ComBat-seq

### Cross-validation with the GLM engines

Both engines satisfy the inductive contract, but two details are worth knowing before dropping them into a `Pipeline`.

**Set a `reference_batch`.** With the default cohort-average target, the batch-free distribution is a size-weighted average across the batches present at fit, so it shifts slightly from fold to fold. A `reference_batch` pins the target *mean* to one batch and removes that dependence. The second moment follows the reference implementations, which differ for a reason. `met`'s target precision is size-weighted like its mean, so it drifts with composition and is pinned as well. `seq`'s target dispersion is an *unweighted* average across batches: it does not drift with batch proportions, and it is already the lower-variance estimate. Pinning it to the reference batch would swap an average of `k` per-batch estimates for one noisier estimate, and measures less CV-stable, so it is deliberately left alone.

**Filter features once, up front.** The pass-through mask is frozen per fold, so a feature that is all-zero in one training fold's batch is corrected in some folds and not in others. Filtering low-count or constant features on the full feature list before cross-validation removes the inconsistency without leaking anything, since the filter uses no outcome information.

### Cost

The GLM engines are **orders of magnitude slower than the Gaussian ones**, and that is worth planning around rather than discovering mid-loop. The dominant cost is the dispersion (or precision) estimation: an adjusted profile likelihood maximized over an 11-point grid, separately for every feature in every batch, each grid point requiring a full GLM refit. Wall-clock on a single modern core, three batches, no covariates:

| Engine | Samples | Features | `fit_transform` | `transform` alone |
|---|---:|---:|---:|---:|
| `seq` | 60 | 2 000 | 0.5 s | 0.3 s |
| `seq` | 120 | 5 000 | 2.6 s | 1.4 s |
| `seq` | 200 | 20 000 | 19 s | 6 s |
| `met` | 60 | 2 000 | 4.5 s | 0.1 s |
| `met` | 120 | 5 000 | 22 s | 0.7 s |
| `met` | 120 | 20 000 | 113 s | 0.7 s |

For scale, Gaussian `ComBat(method="fortin")` on 120 x 5 000 takes about 0.04 s - roughly 60x faster than `seq` and 500x faster than `met` on the same shape. Cost grows about linearly in features and in the number of batches.

Two practical consequences:

- **`transform` is much cheaper than `fit`**, especially for `met`, where it is a pure quantile map with no fitting at all. Under cross-validation you pay the fit cost once per fold and almost nothing per prediction.
- **Filter features first.** Beyond the fold-consistency argument above, dropping uninformative features is the single most effective way to cut the cost, because time is very nearly linear in the feature count.

```{note}
**What is and is not externally validated.** The whole-cohort `fit_transform` path is checked against InMoose `pycombat_seq` (bit-exact) and against fixtures generated from the ComBat-met R package. No implementation of the *inductive* path exists anywhere, so there is nothing to compare `transform` against. It is instead validated by its defining property - that a held-out sample is corrected identically whether it is transformed alone or alongside other samples - which is asserted directly in the test suite. That gap is the honest cost of being first: the split is what combatlearn contributes, and it is also the part no external reference can confirm.
```

## Transductive ComBat (Whole-Cohort Harmonization)

A method belongs in `combatlearn.transductive.TransductiveComBat` on a **configuration criterion, not a mathematical one**. The test is whether the *requested configuration* can be frozen at fit. Concretely, it cannot when a **condition-of-interest / outcome term is protected in the design**: that term is unavailable at transform time for a new sample, and using it would leak the target into the features. Such a configuration can only be applied to a complete cohort in one pass.

The distinction that matters in practice:

- A condition **known at intake** - tissue type, treatment arm, sex - is just an ordinary covariate. Pass it as `discrete_covariates` / `continuous_covariates` on an inductive estimator: no leakage, cross-validation-safe. For counts and proportions that means [`GLMComBat`](#combat-seq-glmcombat-for-count-data), which will also give you `transform`.
- A condition that **is the supervised target** you will later predict is unavailable at transform and leakage-prone if protected. That is transductive-only.

The same engine can therefore sit on either side of the line, and which side it lands on is decided by *your* study design, not by the mathematics. ComBat-seq with tissue type as a covariate is inductive; ComBat-seq with the responder/non-responder label you are about to build a classifier for is not.

`TransductiveComBat` is a single class with a `method=` selector (mirroring `ComBat`), used as a one-shot `fit_transform`. Calling `transform` on separate held-out data raises, and it is deliberately not a `Pipeline` step.

```{note}
"Transductive" here follows scikit-learn's glossary sense: a method that "is designed to model a specific dataset, but not to apply that model to unseen data" - i.e. whole-cohort, non-inductive, `fit_transform`-only. It is *not* the strictly supervised Vapnik sense of transduction (predicting labels for specific unlabeled points): ComBat is unsupervised and predicts no labels.
```

```{warning}
An earlier version of this page justified the tier differently - as holding methods whose "benefit is realized in-sample", and whose count/methylation engines "have no clean fit/transform form". Both claims were wrong, and the criterion above replaces them. Longitudinal ComBat is inductive under a stricter precondition (see [Preconditions and cross-validation](#preconditions-and-cross-validation)), and ComBat-seq / ComBat-met are a parametric GLM fit followed by a strictly per-observation quantile map, which is exactly a `fit` and a `transform`.
```

### Condition-protected ComBat-seq and ComBat-met

`method="seq"` and `method="met"` are the tier's principal occupants. They are the same engines described above, run in the configuration the reference implementations call `full_mod=TRUE`: the condition of interest is protected in the design so that the batch correction cannot absorb it. This is the standard setup for **differential expression** and **differential methylation**, where the whole point is to test that condition afterwards on the same cohort.

`condition` is required, and the error message says why. The arithmetic is not what differs: a condition enters the design exactly as a covariate does, so the correction is bit-identical to passing the same variable to `GLMComBat` as a covariate. What this class adds is that `transform` then refuses to run, so a model fitted around a protected outcome cannot be applied out-of-sample. With no condition to declare there is nothing to refuse, and `GLMComBat` additionally gives you `transform`.

```python
from combatlearn.transductive import TransductiveComBat

harmonizer = TransductiveComBat(
    batch=batch,
    method="seq",
    condition=responder,               # the term the downstream test is about
    discrete_covariates=sex,           # ordinary covariates, as usual
)
counts_corrected = harmonizer.fit_transform(counts)  # whole cohort, one pass
# harmonizer.transform(counts_test) raises - by design
```

Do **not** use this to prepare features for a classifier of `responder`: protecting the target in the harmonization and then predicting it is circular, and the resulting accuracy is not a real estimate. Use `GLMComBat` there, and accept that some of the condition's signal is removed along with the batch effect.

### Deprecated occupant

`method="longitudinal"` is **deprecated** since v2.4.0 (removed in v3.0.0). It is redundant: on a whole cohort every subject is seen at fit, so [`LongitudinalComBat`](#longitudinal-method-longitudinal-combat) returns bit-identical output from `fit_transform` while additionally supporting `transform`.

```python
# Deprecated - use combatlearn.LongitudinalComBat instead
from combatlearn.transductive import TransductiveComBat

harmonizer = TransductiveComBat(
    batch=batch,
    method="longitudinal",
    subject_id=subject,
)
X_corrected = harmonizer.fit_transform(X)  # whole cohort, one pass
```

Once that deprecation window closes, the tier holds only the condition-protected configurations, which is its principled scope.

## Parametric vs Non-Parametric

The Gaussian engines support both parametric and non-parametric empirical Bayes (CovBat's PC-score step is a plain moment match; `GLMComBat` has no such switch):

**Parametric** (default):
- Faster computation
- Assumes normal distribution
- Recommended for most datasets

**Non-Parametric**:
- The `int.eprior` estimate of `sva::ComBat`: each feature's batch parameters are a likelihood-weighted average of the other features' estimates
- No assumption on the prior distribution of the batch parameters
- Cost grows with the square of the number of features
- Use when parametric assumptions violated

```python
# Parametric (default)
combat = ComBat(batch=batch, method="fortin", parametric=True)

# Non-parametric
combat = ComBat(batch=batch, method="fortin", parametric=False)
```

## Mean-Only Correction

The Gaussian engines support mean-only mode, which corrects batch means but preserves variance (for `chen` and `covbat_gam`, see the note below):

```python
combat = ComBat(
    batch=batch,
    method="fortin",
    mean_only=True  # Only correct means
)
```

**Use when**: You want to preserve variance structure across batches.

For `chen` and `covbat_gam`, `mean_only` applies to the Fortin step only: the covariance step still brings the mean and the variance of the PC scores to the pooled ones, as the reference implementation does.

The mean-only estimate follows Fortin's reference implementation (the full empirical-Bayes update, with the variances then fixed to 1), not `sva::ComBat`, which shrinks the batch means with `n = 1` and unit variances.

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

1. **Raw integer counts (RNA-seq)?** → Use `GLMComBat(method="seq")`
2. **Proportions bounded in 0-1 (methylation beta-values)?** → Use `GLMComBat(method="met")` (or `ComBat` on M-values)
3. **Several batch variables to remove (site + scanner + ...)?** → Use `NestedComBat` (with any of the Gaussian engines below as its per-step `method`)
4. **Repeated measures - the same subjects observed more than once?** → Use `LongitudinalComBat` (for cross-validated prediction on *genuinely new* subjects, use Fortin instead)
5. **No covariates?** → Use Johnson
6. **A continuous covariate has a nonlinear effect (e.g. age)?** → Use GAM (`gam`, or `covbat_gam` when batches also differ in feature covariance)
7. **Have covariates + low/normal dimensionality?** → Use Fortin
8. **Have covariates + batches that differ in feature covariance?** → Use Chen

**One question cuts across all of them**: is the term you want to protect the *supervised target* of the analysis that follows?

- **No** - it is known at intake (tissue type, treatment arm, sex). It is an ordinary covariate; pass it as `discrete_covariates` / `continuous_covariates` on any of the inductive estimators above and keep `transform`.
- **Yes** - you will test or predict it afterwards on this same cohort. Then it belongs in the design only for a whole-cohort differential analysis: use [`TransductiveComBat`](#transductive-combat-whole-cohort-harmonization) with `method="seq"` / `method="met"` and accept that there is no `transform`. Never protect a term you are about to predict and then report cross-validated accuracy on it.

## Next Steps

- See the [API Reference](api) for complete parameter documentation

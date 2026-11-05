# FAQ

Frequently asked questions about combatlearn.

---

## Can I use ComBat in a cross-validation pipeline?

Yes. The `ComBat` class implements scikit-learn's `BaseEstimator` and `TransformerMixin`, so it works with `Pipeline`, `cross_val_score`, and `GridSearchCV`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from combatlearn import ComBat

pipe = Pipeline([
    ("combat", ComBat(batch=batch)),
    ("scaler", StandardScaler()),
])
pipe.fit_transform(X)
```

**Important:** Batch labels are passed at construction, not at `fit()`. This means the same batch vector is used for both fitting and transforming. In a cross-validation setting, ensure your batch labels are properly aligned with the train/test splits.

---

## Which method should I use?

| Method | When to use |
|--------|-------------|
| `'johnson'` | Simple batch correction without covariates. Good default for most cases. |
| `'fortin'` | When you have biological covariates (e.g., age, sex, diagnosis) that should be preserved during correction. |
| `'chen'` | When batch effects also affect the covariance structure of the data, not just means and variances. Extends Fortin with PCA-based covariance correction. |
| `'longitudinal'` | Repeated-measures designs (same subjects measured multiple times). Extends Fortin with a per-subject random intercept. **Deprecated on `ComBat`** - use the dedicated `combatlearn.LongitudinalComBat` transformer. For cross-validated prediction on genuinely *new* subjects, prefer `'fortin'`. |
| `'gam'` | When a continuous covariate has a nonlinear effect on the features (e.g., age over the lifespan). Extends Fortin by modeling the covariates in `smooth_terms` with B-splines. Inductive and cross-validation-safe. |
| `'covbat_gam'` | The `'gam'` nonlinear covariate model combined with CovBat's PCA-based covariance correction. |

Start with `'johnson'` if you have no covariates, or `'fortin'` if you do. Use `'chen'` for covariance-level batch effects, `'gam'`/`'covbat_gam'` when a continuous covariate (such as age) acts nonlinearly, and `LongitudinalComBat` for repeated-measures designs (but `'fortin'` for cross-validated prediction that must generalise to new subjects).

Those all assume Gaussian data. If your data are **raw integer counts** or **proportions bounded in 0-1**, they are the wrong family: use the separate `combatlearn.GLMComBat` transformer with `method='seq'` (ComBat-seq, negative binomial) or `method='met'` (ComBat-met, beta regression). Both are inductive too - see [Can I use ComBat-seq / ComBat-met in a cross-validation pipeline?](#faq-glm-cv).

`method` also accepts literature aliases (case- and separator-insensitive): `classic_combat`, `neurocombat`, `covbat`, `longcombat`, `combat_gam`, `covbatgam`, `combat_seq`, `combat_met`.

---

## Parametric or non-parametric?

- **Parametric** (`parametric=True`, default): Assumes batch effect parameters follow specific distributions (inverse gamma for variance, normal for mean). Faster and works well for most omics data.
- **Non-parametric** (`parametric=False`): Puts no parametric prior on the batch effects: each feature's estimate is a likelihood-weighted average of the other features' estimates (`int.eprior` of `sva::ComBat`). Use this when the batch effects do not follow the normal / inverse-gamma priors. Its cost grows with the square of the number of features.

In practice, parametric mode works well for the vast majority of cases.

---

(faq-longitudinal-basics)=

## What is Longitudinal ComBat and when do I need `subject_id` / `time_covariate`?

Use `combatlearn.LongitudinalComBat` (Beer et al., 2020) when the same subjects are measured more than once - across visits, time points, or sites. It extends the Fortin model with a per-subject **random intercept**, estimated by REML, so within-subject correlation is accounted for rather than treated as independent noise.

- **`subject_id`** (required): one label per sample identifying the subject. It drives the random intercept and, like `batch`, is passed at construction and sliced by index, so it stays aligned across train/test splits.
- **`time_covariate`** (optional): a continuous time variable added to the fixed-effects design to model the within-subject trajectory.

By default (`on_unseen_subject="population"`) subjects that were not seen during `fit` are corrected with a zero random intercept - the empirical-Bayes posterior mean for a subject with no observations - so `transform` never fails on new subjects. Pass `on_unseen_subject="error"` to raise instead, listing the unseen subjects.

The older entry points `ComBat(method="longitudinal")` and `TransductiveComBat(method="longitudinal")` are both deprecated and removed in v3.0.0; the algorithm is identical.

---

(faq-longitudinal-cv)=

## Should I use Longitudinal ComBat inside a cross-validation / prediction pipeline?

It depends on what your split does with subjects - and the honest answer needs two things kept apart, because conflating them produced the older advice that this method was whole-cohort-only.

`LongitudinalComBat` is **inductive**, under a precondition one step stricter than `ComBat`'s. `ComBat` needs the **batch** to have been seen at fit; `LongitudinalComBat` needs the **batch *and* the subject** to have been seen at fit. Every parameter, including each subject's random intercept, is estimated at `fit` and frozen, so **the transform itself is leakage-free in every split** - subject *j*'s intercept comes from *j*'s training visits only.

What changes across splits is whether it *helps*, and whether your *benchmark* is trustworthy:

- **Genuinely new test subjects** (subject-grouped CV, `GroupKFold` on `subject_id` - the clinically valid split). The random intercept is zero, so the correction reduces toward Fortin with scale parameters calibrated on subject-centred residuals applied to data that still carries the between-subject spread. It offers little over Fortin here and can harmonise slightly worse. **Prefer `ComBat(method="fortin")`.**
- **Record-wise CV** (one subject's visits split across train and test). The intercept is available and the transform is still leakage-free, but the *evaluation* is biased - by the **split**, not by the harmoniser. Saeb et al. (2017, *GigaScience* 6(5), doi:[10.1093/gigascience/gix019](https://doi.org/10.1093/gigascience/gix019)) show record-wise CV substantially overestimates accuracy through subject-identity confounding with or without a harmoniser in the pipeline. Fix the split.
- **No held-out set at all** (deployment / incremental harmonisation). This is where being inductive pays: fit once on a reference cohort, then `transform` later visits of subjects already in it, without re-correcting the existing data.

See [what `subject_id` does](#faq-longitudinal-basics) for the parameters, and [applying ComBat to new data](#faq-new-unseen-data) for how the batch-level precondition is enforced.

---

## What are `method="gam"` / `method="covbat_gam"` and the spline parameters?

Use the GAM methods (Pomponio et al., 2020) when a continuous covariate has a **nonlinear** effect on the features - the textbook case is age over the lifespan. `method="gam"` is Fortin with the covariates listed in `smooth_terms` modeled by B-splines instead of linearly; `method="covbat_gam"` adds CovBat's covariance correction on top. They are most valuable when the nonlinear covariate is also correlated with batch, where a linear model would absorb part of the covariate's nonlinear effect into the batch correction.

- **`smooth_terms`** (optional): which continuous covariates to model nonlinearly, by column name or integer position. Default `None` smooths every continuous covariate. A term needs at least `spline_df` distinct values.
- **`spline_df`** (default 10): B-spline degrees of freedom (basis functions) per term.
- **`spline_degree`** (default 3): B-spline degree; 3 is cubic.
- **`smooth_term_bounds`** (optional): boundary knots, as a single `(lo, hi)` for all terms or a `{term: (lo, hi)}` dict. Set them wide enough to cover any held-out data; they must contain the training range.

The GAM methods are inductive and cross-validation-safe under the same precondition as Fortin/Chen (batch seen at fit): knots are learned on the training data and reused at transform, and held-out values beyond the boundary knots are clamped (constant extrapolation), so `transform` never fails on out-of-range covariates.

---

## What does `mean_only` do?

When `mean_only=True`, ComBat only adjusts the **location** (mean) of each batch, leaving the **scale** (variance) unchanged. For `'chen'` and `'covbat_gam'` this applies to the ComBat step only: the covariance step still harmonizes the variances of the PC scores. This is useful when:

- You believe batch effects only shift the means.
- You want to preserve the original variance structure of your data.
- Your variance estimates are unreliable due to small sample sizes.

---

(faq-glm-cv)=

## Can I use ComBat-seq / ComBat-met in a cross-validation pipeline?

Yes. `GLMComBat` is a scikit-learn transformer with a real `fit`/`transform` split, so it drops into a `Pipeline` like any other step:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from combatlearn import GLMComBat

pipe = Pipeline([
    ("combat", GLMComBat(batch=batch, method="seq", reference_batch="site_A")),
    ("clf", LogisticRegression()),
])
```

This is the point of the estimator, and it is worth being clear about why it is possible: ComBat-seq and ComBat-met are a parametric GLM fit followed by a **strictly per-observation** quantile map. The corrected value of a sample depends only on that sample's own measurements, batch label, covariates and (for `'seq'`) library size. Everything else is frozen at `fit`. The published implementations are whole-cohort functions, but that is an engineering choice in code written for differential expression, not a property of the method.

Two things to do differently under cross-validation:

- **Set `reference_batch`.** The default target is a size-weighted cohort average, which shifts slightly with fold composition. A reference batch pins the target *mean* to a single batch. It pins whichever target components drift with fold composition, which differs by engine and matches the reference implementations: `met`'s target precision is size-weighted like its mean, so it is pinned too, while `seq`'s target dispersion is an *unweighted* average across batches that does not move with batch proportions. Pinning the `seq` dispersion as well measures *less* stable across folds, not more - averaging `k` per-batch estimates beats adopting any single one.
- **Filter features before the loop, not inside it.** `'seq'` freezes a pass-through mask of features that are all-zero within some batch, and `'met'` one of features the design reproduces exactly. Those masks are fold-dependent, so the same feature can be corrected in one fold and passed through in another. Filtering low-count or constant features up front removes the inconsistency and leaks nothing, since the filter uses no outcome information.

One caveat stated plainly: the whole-cohort `fit_transform` path is verified against InMoose `pycombat_seq` and the ComBat-met R package, but **no implementation of the inductive path exists anywhere else**, so there is no external reference for `transform`. It is validated instead by its defining property - a held-out sample is corrected identically alone or in company - which the test suite asserts directly. See [When do I need the transductive seq/met variants?](#faq-transductive-glm).

---

(faq-transductive-glm)=

## When do I need the transductive seq/met variants?

Only when the term you want to protect **is the supervised target** of the analysis that follows on the same cohort - the differential-expression and differential-methylation setting, where you harmonize and then test that term.

The rule is about your study design, not the engine:

| The condition is... | Example | Use |
|---|---|---|
| Known at intake | tissue type, treatment arm, sex | `GLMComBat(discrete_covariates=...)` - inductive, keeps `transform` |
| The thing you will test on this cohort | case/control in a DE analysis | `TransductiveComBat(method="seq", condition=...)` - `fit_transform` only |
| The thing you will *predict* | responder label for a classifier | `GLMComBat`, **without** protecting it |

```python
from combatlearn.transductive import TransductiveComBat

harmonizer = TransductiveComBat(batch=batch, method="seq", condition=case_control)
counts_corrected = harmonizer.fit_transform(counts)   # transform() raises
```

`condition` is required because declaring the outcome term is what makes the configuration whole-cohort - not because the maths changes. A condition enters the design exactly as a covariate does, so the correction is bit-identical to `GLMComBat` with that variable as a covariate; what this class adds is that `transform` then refuses to run. With no condition to declare there is nothing to refuse, and `GLMComBat` additionally supports `transform`. The third row is the trap - protecting a label in the harmonization and then reporting cross-validated accuracy on that same label is circular, and the accuracy is not a real estimate.

---

(faq-new-unseen-data)=

## Can I apply ComBat to new/unseen data?

Yes, via `transform()`. After fitting on your training data, you can transform new data from the **same batches**:

```python
combat = ComBat(batch=batch_train).fit(X_train)
X_test_corrected = combat.transform(X_test)
```

However, ComBat **cannot** handle batches that were not seen during fitting. If `X_test` contains samples from a new batch, you must re-fit the model with data from all batches.

For `LongitudinalComBat`, unseen *subjects* are handled gracefully by default: a subject absent at fit time is corrected with a zero random intercept (the population correction), or rejected outright with `on_unseen_subject="error"`. Unseen *batches* still raise an error, as above - that precondition is shared by every estimator in the inductive tier.

---

## How do I interpret the `summary()` output?

After fitting, call `summary(combat)` (from `combatlearn.inspection`) for a diagnostic report. Pass `summary(combat, X)` to also report the batch variance explained *after* correcting `X` (equivalently, call `batch_variance_explained(combat, X)` for that number alone). Key sections:

- **Method/Parametric/Mean only**: Confirms your configuration.
- **Samples per batch**: Check for small or imbalanced batches.
- **Top 5 features by batch effect**: Features most affected by batch - useful for quality control.
- **Diagnostics table**:
  - **Batch var. explained (before/after)**: Fraction of total variance explained by batch. Should decrease substantially after correction. The after value appears only when you pass `X` to `summary`.
  - **Design matrix condition number**: Large values (>100) suggest collinearity issues (Fortin/Chen/Longitudinal only).
  - **EB convergence**: For parametric EB, whether the iterative estimation converged for each batch; the non-parametric estimate is reported as "closed form".

---

## How do I choose `covbat_cov_thresh`?

This parameter applies to `method='chen'` (CovBat) and `method='covbat_gam'`, also inside `NestedComBat`. It controls how many principal components are used for covariance correction:

- **Float (0, 1]**: Cumulative variance ratio. `0.9` (default) retains PCs explaining 90% of variance. Higher values correct more subtle covariance effects but risk overfitting.
- **Int >= 1**: Fixed number of PCs, capped at the number of PCs with variance on the training data. Useful when you know the dimensionality of your data's covariance structure.

Start with the default `0.9`. If correction seems insufficient, try `0.95` or `0.99`.

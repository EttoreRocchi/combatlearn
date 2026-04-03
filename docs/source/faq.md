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

Start with `'johnson'` if you have no covariates, or `'fortin'` if you do. Use `'chen'` only if you have evidence of covariance-level batch effects.

---

## Parametric or non-parametric?

- **Parametric** (`parametric=True`, default): Assumes batch effect parameters follow specific distributions (inverse gamma for variance, normal for mean). Faster and works well for most omics data.
- **Non-parametric** (`parametric=False`): Makes no distributional assumptions. Use this when your data violates the parametric assumptions (e.g., heavy-tailed distributions, small sample sizes).

In practice, parametric mode works well for the vast majority of cases.

---

## What does `mean_only` do?

When `mean_only=True`, ComBat only adjusts the **location** (mean) of each batch, leaving the **scale** (variance) unchanged. This is useful when:

- You believe batch effects only shift the means.
- You want to preserve the original variance structure of your data.
- Your variance estimates are unreliable due to small sample sizes.

---

## Can I apply ComBat to new/unseen data?

Yes, via `transform()`. After fitting on your training data, you can transform new data from the **same batches**:

```python
combat = ComBat(batch=batch_train).fit(X_train)
X_test_corrected = combat.transform(X_test)
```

However, ComBat **cannot** handle batches that were not seen during fitting. If `X_test` contains samples from a new batch, you must re-fit the model with data from all batches.

---

## How do I interpret the `summary()` output?

After fitting, call `combat.summary()` for a diagnostic report. Key sections:

- **Method/Parametric/Mean only**: Confirms your configuration.
- **Samples per batch**: Check for small or imbalanced batches.
- **Top 5 features by batch effect**: Features most affected by batch - useful for quality control.
- **Diagnostics table**:
  - **Batch var. explained (before/after)**: Fraction of total variance explained by batch. Should decrease substantially after correction.
  - **Design matrix condition number**: Large values (>100) suggest collinearity issues (Fortin/Chen only).
  - **EB convergence**: Whether the iterative estimation converged for each batch.

---

## How do I choose `covbat_cov_thresh`?

This parameter only applies to `method='chen'` (CovBat). It controls how many principal components are used for covariance correction:

- **Float (0, 1]**: Cumulative variance ratio. `0.9` (default) retains PCs explaining 90% of variance. Higher values correct more subtle covariance effects but risk overfitting.
- **Int >= 1**: Fixed number of PCs. Useful when you know the dimensionality of your data's covariance structure.

Start with the default `0.9`. If correction seems insufficient, try `0.95` or `0.99`.

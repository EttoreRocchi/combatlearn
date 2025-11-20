# Cross-Validation Guide

Learn how to properly use ComBat with cross-validation to prevent data leakage.

## The Data Leakage Problem

**Wrong Approach** ❌:
```python
# DON'T DO THIS!
X_corrected = ComBat(batch=batch).fit_transform(X)
scores = cross_val_score(classifier, X_corrected, y, cv=5)
```

**Problem**: ComBat uses information from ALL samples (including test folds) during correction.

**Right Approach** ✅:
```python
from sklearn.pipeline import Pipeline

# DO THIS!
pipe = Pipeline([
    ("combat", ComBat(batch=batch)),
    ("classifier", classifier)
])
scores = cross_val_score(pipe, X, y, cv=5)
```

**Why it works**: ComBat is fitted on training folds only, then applied to test folds.

## Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("combat", ComBat(batch=batch, method="fortin",
                     continuous_covariates=age)),
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"Mean AUC: {scores.mean():.3f}")
```

## Grid Search with Cross-Validation

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "combat__method": ["johnson", "fortin"],
    "combat__mean_only": [True, False],
    "classifier__C": [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X, y)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
```

## Custom Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
```

## Batch-Aware Cross-Validation

When batches align with outcome labels, use stratified CV:

```python
from sklearn.model_selection import StratifiedKFold

# Ensure balanced batches and labels across folds
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv)
```

## Complete Example

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from combatlearn import ComBat

# Load data
X = pd.read_csv("features.csv", index_col=0)
y = pd.read_csv("labels.csv", index_col=0).squeeze()
batch = pd.read_csv("batch.csv", index_col=0).squeeze()
age = pd.read_csv("age.csv", index_col=0)

# Create pipeline
pipe = Pipeline([
    ("combat", ComBat(batch=batch, continuous_covariates=age)),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(random_state=42))
])

# Define parameter grid
param_grid = {
    "combat__method": ["johnson", "fortin"],
    "combat__parametric": [True, False],
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [5, 10, None]
}

# Grid search with stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc",
                    n_jobs=-1, verbose=1)
grid.fit(X, y)

# Results
print(f"Best CV AUC: {grid.best_score_:.3f}")
print(f"Best params: {grid.best_params_}")
```

## Best Practices

1. **Always use pipelines** when combining ComBat with ML models
2. **Don't fit ComBat before CV** - let the pipeline handle it
3. **Include ComBat params in grid search** to optimize correction
4. **Use stratified CV** when batches correlate with outcomes
5. **Check for batch-outcome correlation** before modeling

## Next Steps

- See [Visualization](visualization.md) guide
- Try [Examples](../examples/cross-validation.md)

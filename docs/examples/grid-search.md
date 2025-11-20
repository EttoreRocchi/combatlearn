# Grid Search Example

This example demonstrates hyperparameter tuning with GridSearchCV.

See the full script: [`examples/grid_search.py`](https://github.com/EttoreRocchi/combatlearn/blob/main/examples/grid_search.py)

## Overview

This example shows how to:
- Include ComBat parameters in grid search
- Optimize batch correction alongside model hyperparameters
- Use cross-validation properly with ComBat

## Key Concepts

### Parameter Prefixes

When ComBat is in a pipeline, prefix parameters with `combat__`:

```python
param_grid = {
    "combat__method": ["johnson", "fortin"],
    "combat__mean_only": [True, False],
    "classifier__C": [0.1, 1.0, 10.0]
}
```

## Running the Example

```bash
python examples/grid_search.py
```

## Code Example

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from combatlearn import ComBat

# Create pipeline
pipe = Pipeline([
    ("combat", ComBat(batch=batch, method="fortin",
                     continuous_covariates=age)),
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

# Define grid
param_grid = {
    "combat__method": ["johnson", "fortin"],
    "combat__parametric": [True, False],
    "combat__mean_only": [True, False],
    "classifier__C": [0.1, 1.0, 10.0],
    "classifier__kernel": ["linear", "rbf"]
}

# Grid search
grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy")
grid.fit(X, y)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
```

## Results Analysis

The example also shows how to analyze the impact of different parameters:

```python
results = pd.DataFrame(grid.cv_results_)

# Group by Combat method
method_scores = results.groupby("param_combat__method")["mean_test_score"].agg(["mean", "std"])
print(method_scores)
```

## Next Steps

- Learn about [Cross-Validation](../user-guide/cross-validation.md)
- See [Visualization Guide](../user-guide/visualization.md)

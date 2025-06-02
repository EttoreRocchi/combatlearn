# **combatlearn**

<div align="left">
<p><img src="docs/logo.png" width="350" /></p>
</div>

**combatlearn** makes the popular _ComBat_ batch-effect correction algorithm available for use into machine learning frameworks. It lets you harmonise high-dimensional data inside a scikit-learn `Pipeline`, so that cross-validation and grid-search automatically take batch structure into account, **without data leakage**.

**Three methods**:
- `method="johnson"` - classic ComBat (Johnson _et al._, 2007)
- `method="fortin"` - covariate-aware ComBat (Fortin _et al._, 2018)
- `method="chen"` - CovBat (Chen _et al._, 2022)

## Installation

```bash
git clone https://github.com/EttoreRocchi/comabtlearn.git
cd combatlearn
pip install .
pytest -q # optional, to run tests
```

## Quick start

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from combatlearn import ComBat

df = pd.read_csv("data.csv", index_col=0)
X, y = df.drop(columns="y"), df["y"]

batch = pd.read_csv("batch.csv", index_col=0, squeeze=True)
diag = pd.read_csv("diagnosis.csv", index_col=0) # categorical
age = pd.read_csv("age.csv", index_col=0) # continuous

pipe = Pipeline([
    ("combat", ComBat(
        batch=batch,
        discrete_covariates=diag,
        continuous_covariates=age,
        method="fortin", # or "johnson" or "chen"
        parametric=True # default
    )),

    ("clf", LogisticRegression())
])

param_grid = {
    "combat__parametric": [True, False],
    "clf__C": [0.01, 0.1, 1, 10],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
)

grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print(f"Best CV AUROC: {grid.best_score_:.3f}")
```

## Contributing

Pull requests, bug reports, and feature ideas are welcome: feel free to open a PR!

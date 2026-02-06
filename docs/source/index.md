# combatlearn

[![Python versions](https://img.shields.io/badge/python-%3E%3D3.10-blue?logo=python)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/combatlearn?cacheSeconds=300)](https://pypi.org/project/combatlearn/)
[![License](https://img.shields.io/github/license/EttoreRocchi/combatlearn)](https://github.com/EttoreRocchi/combatlearn/blob/main/LICENSE)
[![Test](https://github.com/EttoreRocchi/combatlearn/actions/workflows/test.yaml/badge.svg)](https://github.com/EttoreRocchi/combatlearn/actions/workflows/test.yaml)

```{image} _static/logo.png
:alt: combatlearn logo
:width: 350px
:align: center
```

**combatlearn** makes the popular _ComBat_ (and _CovBat_) batch-effect correction algorithm available for use in machine learning frameworks. It lets you harmonize high-dimensional data inside a scikit-learn `Pipeline`, so that cross-validation and grid-search automatically take batch structure into account, **without data leakage**.

## Features

- **Three ComBat Methods**:
  - `method="johnson"` - Classic ComBat (Johnson et al., 2007)
  - `method="fortin"` - neuroCombat with covariates (Fortin et al., 2018)
  - `method="chen"` - CovBat PCA-based (Chen et al., 2022)

- **Scikit-learn Compatible**:
  - Works seamlessly in `Pipeline` objects
  - Compatible with `GridSearchCV` and `cross_val_score`
  - Prevents data leakage during cross-validation

- **Visualization Tools**:
  - Built-in plotting with PCA, t-SNE, and UMAP
  - Static (matplotlib) and interactive (plotly) visualizations
  - Before/after batch effect comparison

- **Feature Importance Analysis** *(New in v1.2.0)*:
  - Identify which features have strongest batch effects
  - Location (mean shift) and scale (variance) decomposition
  - Magnitude and distribution modes for different use cases

## Quick Example

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from combatlearn import ComBat

# Load your data
X = pd.read_csv("data.csv", index_col=0)
y = pd.read_csv("labels.csv", index_col=0).squeeze()
batch = pd.read_csv("batch.csv", index_col=0).squeeze()

# Create pipeline with ComBat
pipe = Pipeline([
    ("combat", ComBat(batch=batch, method="fortin")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

# Hyperparameter tuning with grid search
param_grid = {
    "combat__mean_only": [True, False],
    "clf__C": [0.01, 0.1, 1, 10],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid.fit(X, y)

print(f"Best CV AUROC: {grid.best_score_:.3f}")
```

## Why combatlearn?

Batch effects are systematic technical variations that can confound biological signals in high-dimensional data. ComBat is the gold standard for batch effect correction, but traditional implementations don't integrate well with machine learning workflows.

**combatlearn solves this by**:

- Fitting ComBat parameters on training data only
- Applying the same transformation to test data
- Preventing data leakage in cross-validation
- Supporting hyperparameter tuning of batch correction

## Installation

```bash
pip install combatlearn
```

## Citation

If combatlearn is useful in your research, please cite the original ComBat papers:

- **Johnson et al. (2007)**: [Adjusting batch effects in microarray expression data using empirical Bayes methods](https://doi.org/10.1093/biostatistics/kxj037). *Biostatistics*, 8(1):118-27.

- **Fortin et al. (2018)**: [Harmonization of cortical thickness measurements across scanners and sites](https://doi.org/10.1016/j.neuroimage.2017.11.024). *Neuroimage*, 167:104-120.

- **Chen et al. (2022)**: [Mitigating site effects in covariance for machine learning in neuroimaging data](https://doi.org/10.1002/hbm.25688). *Hum Brain Mapp*, 43(4):1179-1195.

## Author

**Ettore Rocchi** @ University of Bologna

[Google Scholar](https://scholar.google.com/citations?user=MKHoGnQAAAAJ) | [Scopus](https://www.scopus.com/authid/detail.uri?authorId=57220152522) | [GitHub](https://github.com/EttoreRocchi)

## License

MIT License - see [LICENSE](https://github.com/EttoreRocchi/combatlearn/blob/main/LICENSE) for details.

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

methods
api
contributing
```

# **combatlearn**

[![Python versions](https://img.shields.io/badge/python-%3E%3D3.10-blue?logo=python)](https://www.python.org/)
[![Test](https://github.com/EttoreRocchi/combatlearn/actions/workflows/test.yaml/badge.svg)](https://github.com/EttoreRocchi/combatlearn/actions/workflows/test.yaml)
[![PyPI Downloads](https://static.pepy.tech/badge/combatlearn)](https://pepy.tech/projects/combatlearn)
[![PyPI Version](https://img.shields.io/pypi/v/combatlearn?cacheSeconds=300)](https://pypi.org/project/combatlearn/)
[![License](https://img.shields.io/github/license/EttoreRocchi/combatlearn)](https://github.com/EttoreRocchi/combatlearn/blob/main/LICENSE)

<div align="center">
<p><img src="https://raw.githubusercontent.com/EttoreRocchi/combatlearn/main/docs/logo.png" alt="combatlearn logo" width="350" /></p>
</div>

**combatlearn** makes the popular _ComBat_ (and _CovBat_) batch-effect correction algorithm available for use into machine learning frameworks. It lets you harmonise high-dimensional data inside a scikit-learn `Pipeline`, so that cross-validation and grid-search automatically take batch structure into account, **without data leakage**.

**Three methods**:
- `method="johnson"` - classic ComBat (Johnson _et al._, 2007)
- `method="fortin"` - neuroComBat (Fortin _et al._, 2018)
- `method="chen"` - CovBat (Chen _et al._, 2022)

## Installation

```bash
pip install combatlearn
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
        parametric=True
    )),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

param_grid = {
    "combat__mean_only": [True, False],
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

For a full example of how to use **combatlearn** see the [notebook demo](https://github.com/EttoreRocchi/combatlearn/blob/main/demo/combatlearn_demo.ipynb)

## `ComBat` parameters

The following section provides a detailed explanation of all parameters available in the scikit-learn-compatible `ComBat` class.

### Main Parameters

| Parameter | Type | Default | Description |
| --- | ---  | --- | --- |
| `batch` | array-like or pd.Series | **required** | Vector indicating batch assignment for each sample. This is used to estimate and remove batch effects. |
| `discrete_covariates` | array-like, pd.Series, or pd.DataFrame | `None` | Optional categorical covariates (e.g., sex, site). Only used in `"fortin"` and `"chen"` methods. |
| `continuous_covariates` | array-like, pd.Series or pd.DataFrame | `None` | Optional continuous covariates (e.g., age). Only used in `"fortin"` and `"chen"` methods. |

### Algorithm Options

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `method` | str | `"johnson"` | ComBat method to use: <ul><li>`"johnson"` - Classical ComBat (_Johnson et al. 2007_)</li><li>`"fortin"` - ComBat with covariates (_Fortin et al. 2018_)</li><li>`"chen"` - CovBat, PCA-based correction (_Chen et al. 2022_)</li></ul> |
| `parametric` | bool | `True` | Whether to use the **parametric empirical Bayes** formulation. If `False`, a non-parametric iterative scheme is used. |
| `mean_only` | bool | `False` | If `True`, only the **mean** is corrected, while variances are left unchanged. Useful for preserving variance structure in the data. |
| `reference_batch` | str or `None` | `None` | If specified, acts as a reference batch - other batches will be corrected to match this one. |
| `covbat_cov_thresh` | float, int | `0.9` | For `"chen"` method only: Cumulative variance threshold $]0,1[$ to retain PCs in PCA space (e.g., 0.9 = retain 90% explained variance). If an integer is provided, it represents the number of principal components to use. |
| `eps` | float | `1e-8` | Small jitter value added to variances to prevent divide-by-zero errors during standardization. |

## Contributing

Pull requests, bug reports, and feature ideas are welcome: feel free to open a PR!

## Author

[**Ettore Rocchi**](https://github.com/ettorerocchi) @ University of Bologna

[Google Scholar](https://scholar.google.com/citations?user=MKHoGnQAAAAJ) $\cdot$ [Scopus](https://www.scopus.com/authid/detail.uri?authorId=57220152522)

## Acknowledgements

This project builds on the excellent work of the ComBat family of harmonisation methods.
We gratefully acknowledge:

- [**ComBat**](https://rdrr.io/bioc/sva/man/ComBat.html)
- [**neuroCombat**](https://github.com/Jfortin1/neuroCombat)
- [**CovBat**](https://github.com/andy1764/CovBat_Harmonization)

## Citation

If **combatlearn** is useful in your research, please cite the original
papers:

- Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. _Biostatistics_. 2007 Jan;8(1):118-27. doi: [10.1093/biostatistics/kxj037](https://doi.org/10.1093/biostatistics/kxj037)

- Fortin JP, Cullen N, Sheline YI, Taylor WD, Aselcioglu I, Cook PA, Adams P, Cooper C, Fava M, McGrath PJ, McInnis M, Phillips ML, Trivedi MH, Weissman MM, Shinohara RT. Harmonization of cortical thickness measurements across scanners and sites. _Neuroimage_. 2018 Feb 15;167:104-120. doi: [10.1016/j.neuroimage.2017.11.024](https://doi.org/10.1016/j.neuroimage.2017.11.024)

- Chen AA, Beer JC, Tustison NJ, Cook PA, Shinohara RT, Shou H; Alzheimer's Disease Neuroimaging Initiative. Mitigating site effects in covariance for machine learning in neuroimaging data. _Hum Brain Mapp_. 2022 Mar;43(4):1179-1195. doi: [10.1002/hbm.25688](https://doi.org/10.1002/hbm.25688)
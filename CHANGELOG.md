# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.4.0] - 2026-11-05

### Added

- **GLMComBat** (`combatlearn.GLMComBat`): scikit-learn transformer for non-Gaussian data, inductive and cross-validation-safe like `ComBat`.
  - `method="seq"`: **ComBat-seq** (_Zhang, Parmigiani & Johnson_ 2020) for raw RNA-seq counts; integer in, integer out.
  - `method="met"`: **ComBat-met** (_Wang_ 2025) for DNA-methylation beta-values; output stays inside `(0, 1)`.
  - Covariates via `discrete_covariates` / `continuous_covariates`, as on `ComBat`; `reference_batch` is recommended inside cross-validation.
  - Matches InMoose `pycombat_seq` bit-identically and the ComBat-met R package to within 1e-6.
- **LongitudinalComBat** (`combatlearn.LongitudinalComBat`): inductive Longitudinal ComBat (_Beer et al._ 2020) with `transform` support. Requires `subject_id`, accepts `time_covariate`; subjects unseen at fit get the population correction, or raise with `on_unseen_subject="error"`.
- **`TransductiveComBat(method="seq"/"met")`**: condition-protected ComBat-seq and ComBat-met for differential-expression and differential-methylation workflows, via the new `condition` parameter (required for these methods). `fit_transform` only.

### Changed

- **Longitudinal ComBat is inductive**: cross-validation-safe like `ComBat`, provided both the batch and the subject were seen at fit; otherwise it falls back to the population correction.
- **`combatlearn.transductive`** holds only the configurations that protect a condition of interest, which cannot be frozen at fit.
- **The `ComBat(method="longitudinal")` deprecation** now points to `combatlearn.LongitudinalComBat`.
- **`ComBat(method="seq"/"met")`** raises and points to `GLMComBat`.
- **A discrete covariate that determines `y` warns at `fit`** on `ComBat`, `LongitudinalComBat`, `NestedComBat` and `GLMComBat`, pointing to `TransductiveComBat`.
- **CovBat** harmonizes the PC scores without empirical-Bayes shrinkage, as in _Chen et al._ 2022.
- **`mean_only=True` with `chen` / `covbat_gam`** applies to the ComBat step only; the PC-score variances are still harmonized.
- **`parametric=False`** uses the non-parametric estimate of `sva::ComBat`; its cost grows with the square of the number of features.
- **`summary()`** reports the non-parametric estimate as "closed form".
- **`plot_batch_effect_heatmap()`** uses a color scale symmetric around 0.
- **`compute_batch_metrics()`** documents the direction of each batch metric.
- **SciPy is a declared dependency** (`scipy>=1.8`).

### Deprecated

- **`TransductiveComBat(method="longitudinal")`**: use `combatlearn.LongitudinalComBat`, which gives the same output and also supports `transform`. Removal in v3.0.0.

### Fixed

- **Empirical-Bayes shrinkage** pools each batch's estimates across features, as in _Johnson et al._ 2007 and `sva::ComBat`. Changes the output of every Gaussian engine; `johnson`, `fortin` and `chen` now match the reference R implementation.
- **CovBat preserves protected covariate effects** (`chen`, `covbat_gam`, also inside `NestedComBat`).
- **CovBat brings each batch's PC-score variance to the pooled variance.** Changes every `chen` / `covbat_gam` output.
- **CovBat no longer discards part of held-out samples** when there are fewer training samples than features.
- **CovBat no longer distorts held-out samples** when `covbat_cov_thresh` asks for more PCs than the training data support.
- **`covbat_cov_thresh=1.0` no longer raises.**
- **`reference_batch` maps every other batch onto the reference batch's mean and variance** with `johnson`, `longitudinal`, `chen` and `covbat_gam`, also when batch scales differ or the reference batch is small.
- **Discrete covariates are coded with the levels seen at fit** on `ComBat`, `LongitudinalComBat` and `NestedComBat`, so held-out samples are coded as the training data.
- **A discrete covariate level not seen at fit raises at `transform`** on `ComBat`, `LongitudinalComBat`, `NestedComBat` and `GLMComBat`, including values of another type (such as 0/1 for True/False) and categories absent from the training rows.
- **Null batch labels and covariates raise at `fit` and `transform`** on `ComBat`, `LongitudinalComBat`, `NestedComBat` and `GLMComBat`.

## [2.3.0] - 2026-07-20

### Added

- **NestedComBat** (`combatlearn.NestedComBat`): harmonizes several batch variables at once (e.g. site, scanner, protocol) by applying ComBat to each in sequence (Nested / OPNested / GMM ComBat, _Horng et al._ 2022). Inductive and cross-validation-safe like `ComBat`.
  - Per-step engine via `method` (`"fortin"`, `"chen"`, `"gam"`, `"covbat_gam"`); covariates are preserved across steps.
  - `optimize_order=True` picks the order that minimizes the residual batch effect (exhaustive up to `max_exhaustive_vars`, greedy above).
  - `gmm="batch"` / `"covariate"` adds a latent Gaussian-mixture grouping, harmonized away or preserved as a covariate.
- **`combatlearn.transductive.TransductiveComBat`** (`method="longitudinal"`): whole-cohort, `fit_transform`-only harmonizer; not a `Pipeline` step.
- **`combatlearn.inspection.batch_variance_explained(combat, X)`**: fraction of variance explained by batch after correcting `X`. `summary()` gains an optional `X` argument that adds this value.

### Changed

- **`ComBat.transform` no longer stores the post-correction batch variance**; use `batch_variance_explained(combat, X)` or `summary(combat, X)`.

### Deprecated

- **`ComBat(method="longitudinal")`**: use `TransductiveComBat(method="longitudinal")`. Removal in v3.0.0.

## [2.2.0] - 2026-07-06

### Added

- **ComBat-GAM** (`method="gam"`) and **CovBat-GAM** (`method="covbat_gam"`): continuous covariates modeled nonlinearly with B-splines (_Pomponio et al._ 2020). Inductive and cross-validation-safe like `fortin` / `chen`.
- **New parameters on `ComBat`**: `smooth_terms` (covariates to smooth; default all), `spline_df` (default 10), `spline_degree` (default 3) and `smooth_term_bounds` (optional per-term range).
- **`layout="diverging"`** on `plot_feature_diagnostics(kind="combined")`: location and scale as back-to-back bars with separate axes. The default stays `layout="grouped"`.

### Changed

- **`method` accepts literature aliases**, case- and separator-insensitive: `classic_combat` (johnson), `neurocombat` (fortin), `covbat` (chen), `longcombat` (longitudinal), `combat_gam` (gam), `covbatgam` (covbat_gam).

## [2.1.0] - 2026-06-16

### Added

- **Longitudinal ComBat** (`method="longitudinal"`) for repeated-measures designs, with a per-subject random intercept (_Beer et al._ 2020). New `subject_id` (required) and `time_covariate` parameters.

## [2.0.0] - 2026-04-07

### Added

- **`combatlearn.inspection`**: standalone `feature_batch_diagnostics()` and `summary()` for fitted models.
- **`weighted` parameter** of `feature_batch_diagnostics()` (default `True`): batch contributions weighted by sample size.
- **Faster import**: plotting and embedding libraries are loaded only when used.

### Changed

- **Metrics, inspection and visualization are standalone functions** that take a fitted `ComBat` as first argument; `ComBat` keeps only the scikit-learn methods (`fit`, `transform`, `fit_transform`, `get_feature_names_out`, `get_params`, `set_params`).
- **`feature_batch_importance()`** is renamed `feature_batch_diagnostics()` and lives in `combatlearn.inspection`.
- **`compute_batch_metrics(combat, X, ...)`** lives in `combatlearn.metrics`, **`summary(combat)`** in `combatlearn.inspection`, and `plot_transformation(combat, X, ...)`, `plot_feature_diagnostics(combat, ...)`, `plot_batch_effect_heatmap(combat, ...)` in `combatlearn.visualization`.

  | v1.x | v2.0 |
  |------|------|
  | `cb.compute_batch_metrics(X)` | `compute_batch_metrics(cb, X)` |
  | `cb.feature_batch_importance()` | `feature_batch_diagnostics(cb)` |
  | `cb.summary()` | `summary(cb)` |
  | `cb.plot_transformation(X)` | `plot_transformation(cb, X)` |
  | `cb.plot_feature_diagnostics()` | `plot_feature_diagnostics(cb)` |
  | `cb.plot_batch_effect_heatmap()` | `plot_batch_effect_heatmap(cb)` |
  | `ComBat(batch, compute_metrics=True)` | `metrics = compute_batch_metrics(cb, X)` |
  | `cb.metrics_` | `metrics = compute_batch_metrics(cb, X)` |

### Removed

- **`compute_metrics` parameter** of `ComBat`.
- **`metrics_` property** of `ComBat`.
- **`ComBatMetricsMixin` and `ComBatVisualizationMixin`**.

## [1.4.0] - 2026-04-03

### Added

- **Diagnostics in `summary()`**: batch variance explained, design condition number and empirical-Bayes convergence.
- **Input warnings** for near-zero-variance features, imbalanced batches and collinear covariates.
- **Type hints** shipped with the package (`py.typed`).
- **FAQ** in the documentation.

## [1.3.0] - 2026-03-20

### Added

- **`nn_algorithm` parameter** for the nearest-neighbor metrics (kBET, LISI, k-NN preservation).
- **Input validation** in `fit()` and `transform()`: NaN or Inf in `X`, batch or covariates raise.
- **`summary()`** with an overview of the fitted model.
- **`plot_batch_effect_heatmap()`**.
- **`get_feature_names_out()`**, for `set_output(transform="pandas")` in pipelines.

### Changed

- **seaborn is a dependency** (`seaborn>=0.12`).
- **The demo notebook is part of the documentation.**

### Removed

- **`examples/` scripts**, replaced by the demo notebook.

### Fixed

- **Metrics and feature importance on an unfitted model** raise a clear error.

## [1.2.0] - 2026-02-06

### Added

- **`feature_batch_importance()`**: ranks features by their location (gamma) and scale (delta) batch shift.
- **`plot_feature_importance()`**: per-feature batch-effect magnitudes.

## [1.1.2] - 2026-01-16

### Fixed

- **`method="chen"`**: `transform` now centers held-out data consistently with `fit`.

## [1.1.1] - 2026-01-15

### Changed

- **Documentation moved to Sphinx**, at the same address.

## [1.1.0] - 2026-01-08

### Added

- **`compute_batch_metrics()`**: correction-quality metrics for batch effect (silhouette, Davies-Bouldin, kBET, LISI, variance ratio), structure preservation (k-NN preservation, distance correlation) and alignment (centroid distance, Levene statistic).
- **Python 3.13** support (3.10-3.13).

## [1.0.0] - 2025-11-20

### Added

- **Documentation** at [combatlearn.readthedocs.io](https://combatlearn.readthedocs.io/).

### Changed

- **Development dependencies** moved to the optional `dev` extra.

### Fixed

- **Deprecation warnings** no longer raised during use.

## [0.2.2] - 2025-07-22

### Fixed

- **`plot_transformation()`**: batch colors are consistent across interactive plots.

## [0.2.1] - 2025-07-21

### Fixed

- **matplotlib is a declared dependency**, required by `plot_transformation()`.

## [0.2.0] - 2025-07-21

### Added

- **`plot_transformation()`**: before/after view of the correction with PCA, t-SNE or UMAP, static or interactive.

## [0.1.2] - 2025-07-10

### Added

- **`reference_batch` for `method="fortin"`.**

### Changed

- **`method="fortin"`** follows neuroCombat (_Fortin et al._ 2018) more closely in how covariates and batch effects are estimated; outputs change.

## [0.1.1] - 2025-07-01

### Changed

- **Requires Python 3.10 or later.**
- **More numerically stable estimation** of batch and covariate effects.

## [0.1.0] - 2025-06-13

### Added

- **`ComBat`**, a scikit-learn transformer for batch-effect correction inside a `Pipeline`, without data leakage: `method="johnson"` (_Johnson et al._ 2007), `"fortin"` (_Fortin et al._ 2018) and `"chen"` (CovBat, _Chen et al._ 2022), with `parametric`, `mean_only`, `reference_batch` and `covbat_cov_thresh`.

[2.4.0]: https://github.com/EttoreRocchi/combatlearn/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/EttoreRocchi/combatlearn/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/EttoreRocchi/combatlearn/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/EttoreRocchi/combatlearn/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/EttoreRocchi/combatlearn/compare/v1.4.0...v2.0.0
[1.4.0]: https://github.com/EttoreRocchi/combatlearn/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/EttoreRocchi/combatlearn/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/EttoreRocchi/combatlearn/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/EttoreRocchi/combatlearn/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/EttoreRocchi/combatlearn/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/EttoreRocchi/combatlearn/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/EttoreRocchi/combatlearn/compare/v0.2.2...v1.0.0
[0.2.2]: https://github.com/EttoreRocchi/combatlearn/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/EttoreRocchi/combatlearn/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/EttoreRocchi/combatlearn/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/EttoreRocchi/combatlearn/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/EttoreRocchi/combatlearn/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EttoreRocchi/combatlearn/releases/tag/v0.1.0

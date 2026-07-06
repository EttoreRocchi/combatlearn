# Changelog

All notable changes to combatlearn are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [2.2.0] - 2026-07-06

### Added

- **ComBat-GAM** (`method="gam"`) and **CovBat-GAM** (`method="covbat_gam"`): model continuous covariates nonlinearly (B-splines) instead of linearly. Inductive and cross-validation-safe like `fortin`/`chen`.
- New parameters on `ComBat` / `ComBatModel`: `smooth_terms` (which continuous covariates to smooth; default all of them), `spline_df` (default 10), `spline_degree` (default 3), and `smooth_term_bounds` (optional per-term range).
- **`layout="diverging"`** option on `plot_feature_diagnostics()` (for `kind="combined"`): draws location and scale as back-to-back bars, each with its own x-axis, so a small component stays readable next to a large one. Default remains `layout="grouped"`.

### Changed

- **Method aliases**: `method` now accepts case- and separator-insensitive literature aliases, resolved to the canonical author names: `classic_combat` (johnson), `neurocombat` (fortin), `covbat` (chen), `longcombat` (longitudinal), `combat_gam` (gam), `covbatgam` (covbat_gam).

## [2.1.0] - 2026-06-16

### Added

- **Longitudinal ComBat** (`method="longitudinal"`) for repeated-measures designs. The fixed-effects mean model is fit with a per-subject random intercept (REML) and the random-intercept BLUP is included in the standardization, after which the standard empirical-Bayes harmonization is applied. New `subject_id` (required) and `time_covariate` parameters on `ComBat` / `ComBatModel`.

## [2.0.0] - 2026-04-07

First changelog-tracked release of combatlearn.

### Added

- **`combatlearn.inspection` module**: standalone `feature_batch_diagnostics()` and `summary()` functions for inspecting fitted ComBat models.
- **Weighted RMS** in `feature_batch_diagnostics()`: new `weighted` parameter (default `True`) weights batch contributions by sample size for more representative importance scores.
- **Lazy imports** in `combatlearn.visualization`: matplotlib, plotly, seaborn, umap, and sklearn reducers are imported at function call time, not at module import.

### Changed

- **Architecture refactor: composition over mixins.** `ComBatMetricsMixin` and `ComBatVisualizationMixin` removed. All metrics, inspection, and visualization functions are now standalone, accepting a fitted `ComBat` instance as first argument.
- **`ComBat` class** retains only sklearn-standard methods: `fit`, `transform`, `fit_transform`, `get_feature_names_out`, `set_params`, `get_params`.
- **`feature_batch_importance()` renamed** to `feature_batch_diagnostics()` and moved to `combatlearn.inspection`.
- **`compute_batch_metrics()`** moved to `combatlearn.metrics` as a standalone function: `compute_batch_metrics(combat, X, ...)`.
- **`summary()`** moved to `combatlearn.inspection` as a standalone function: `summary(combat)`.
- **Visualization functions** moved to `combatlearn.visualization` as standalone functions: `plot_transformation(combat, X, ...)`, `plot_feature_diagnostics(combat, ...)`, `plot_batch_effect_heatmap(combat, ...)`.

### Removed

- **`compute_metrics` parameter** from `ComBat.__init__()`.
- **`metrics_` caching property** from `ComBat`.
- **`ComBatMetricsMixin` class**.
- **`ComBatVisualizationMixin` class**.

### Migration

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

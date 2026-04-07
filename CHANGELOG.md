# Changelog

All notable changes to combatlearn are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

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

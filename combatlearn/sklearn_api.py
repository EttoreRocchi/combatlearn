"""Scikit-learn compatible ComBat wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .core import ArrayLike, ComBatModel
from .metrics import ComBatMetricsMixin
from .visualization import ComBatVisualizationMixin


class ComBat(ComBatMetricsMixin, ComBatVisualizationMixin, BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper around `ComBatModel`.

    Stores batch (and optional covariates) passed at construction and
    appropriately uses them for separate `fit` and `transform`.
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        method: str = "johnson",
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: str | None = None,
        eps: float = 1e-8,
        covbat_cov_thresh: float | int = 0.9,
        compute_metrics: bool = False,
    ) -> None:
        self.batch = batch
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.method = method
        self.parametric = parametric
        self.mean_only = mean_only
        self.reference_batch = reference_batch
        self.eps = eps
        self.covbat_cov_thresh = covbat_cov_thresh
        self.compute_metrics = compute_metrics
        self._model = ComBatModel(
            method=method,
            parametric=parametric,
            mean_only=mean_only,
            reference_batch=reference_batch,
            eps=eps,
            covbat_cov_thresh=covbat_cov_thresh,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> ComBat:
        """Fit the ComBat model."""
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        batch_vec = self._subset(self.batch, idx)
        disc = self._subset(self.discrete_covariates, idx)
        cont = self._subset(self.continuous_covariates, idx)
        self._model.fit(
            X,
            batch=batch_vec,
            discrete_covariates=disc,
            continuous_covariates=cont,
        )
        self._fitted_batch = batch_vec
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Transform the data using fitted ComBat parameters."""
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        batch_vec = self._subset(self.batch, idx)
        disc = self._subset(self.discrete_covariates, idx)
        cont = self._subset(self.continuous_covariates, idx)
        return self._model.transform(
            X,
            batch=batch_vec,
            discrete_covariates=disc,
            continuous_covariates=cont,
        )

    @staticmethod
    def _subset(obj: ArrayLike | None, idx: pd.Index) -> pd.DataFrame | pd.Series | None:
        """Subset array-like object by index."""
        if obj is None:
            return None
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.loc[idx]
        else:
            if isinstance(obj, np.ndarray) and obj.ndim == 1:
                return pd.Series(obj, index=idx)
            else:
                return pd.DataFrame(obj, index=idx)

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> pd.DataFrame:
        """
        Fit and transform the data, optionally computing metrics.

        If ``compute_metrics=True`` was set at construction, batch effect
        metrics are computed and cached in the ``metrics_`` property.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit and transform.
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        X_transformed : pd.DataFrame
            Batch-corrected data.
        """
        self.fit(X, y)
        X_transformed = self.transform(X)

        if self.compute_metrics:
            self._metrics_cache = self.compute_batch_metrics(X)

        return X_transformed

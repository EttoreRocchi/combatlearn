"""Scikit-learn compatible ComBat wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .core import ArrayLike, ComBatModel
from .metrics import ComBatMetricsMixin
from .visualization import ComBatVisualizationMixin


class ComBat(ComBatMetricsMixin, ComBatVisualizationMixin, BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper around `ComBatModel`.

    Stores batch (and optional covariates) passed at construction and
    appropriately uses them for separate `fit` and `transform`.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample.
    discrete_covariates : array-like, optional
        Categorical covariates to protect (Fortin/Chen only).
    continuous_covariates : array-like, optional
        Continuous covariates to protect (Fortin/Chen only).
    method : {'johnson', 'fortin', 'chen'}, default='johnson'
        ComBat variant to use.
    parametric : bool, default=True
        Use parametric empirical Bayes.
    mean_only : bool, default=False
        Adjust only the mean (ignore variance).
    reference_batch : str, optional
        Batch level to leave unchanged.
    eps : float, default=1e-8
        Numerical jitter for stability.
    covbat_cov_thresh : float or int, default=0.9
        CovBat variance threshold for PCs.
    compute_metrics : bool, default=False
        If True, ``fit_transform`` caches batch metrics in ``metrics_``.
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

    def _validate_inputs(self, X: ArrayLike, *, fitting: bool = False) -> None:
        """Validate X and, during fitting, batch/covariates for NaN/Inf."""
        check_array(X, ensure_all_finite=True, dtype="numeric")

        if fitting:
            batch_ser = (
                pd.Series(self.batch) if not isinstance(self.batch, pd.Series) else self.batch
            )
            if batch_ser.isna().any():
                raise ValueError("batch contains NaN values.")

            if self.discrete_covariates is not None:
                disc_df = (
                    pd.DataFrame(self.discrete_covariates)
                    if not isinstance(self.discrete_covariates, (pd.Series, pd.DataFrame))
                    else self.discrete_covariates
                )
                if (
                    disc_df.isna().any().any()
                    if isinstance(disc_df, pd.DataFrame)
                    else disc_df.isna().any()
                ):
                    raise ValueError("discrete_covariates contains NaN values.")

            if self.continuous_covariates is not None:
                cont_arr = self.continuous_covariates
                if isinstance(cont_arr, (pd.Series, pd.DataFrame)):
                    cont_arr = cont_arr.values
                check_array(
                    np.atleast_2d(cont_arr) if np.asarray(cont_arr).ndim == 1 else cont_arr,
                    ensure_all_finite=True,
                    dtype="numeric",
                )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> ComBat:
        """Fit the ComBat model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit.
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        self : ComBat
            Fitted estimator.
        """
        self._validate_inputs(X, fitting=True)
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object
            )

        self._model = ComBatModel(
            method=self.method,
            parametric=self.parametric,
            mean_only=self.mean_only,
            reference_batch=self.reference_batch,
            eps=self.eps,
            covbat_cov_thresh=self.covbat_cov_thresh,
        )

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
        """Transform the data using fitted ComBat parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : pd.DataFrame
            Batch-corrected data.
        """
        self._validate_inputs(X)
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

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray:
        """Get output feature names for transform.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Ignored. Present for API compatibility.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Feature names.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the estimator is not fitted.
        """
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_

    def summary(self) -> str:
        """Return a human-readable diagnostic report after fitting.

        Returns
        -------
        str
            Multi-line summary string.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if not hasattr(self, "_model") or not hasattr(self._model, "_gamma_star"):
            raise ValueError("This ComBat instance is not fitted yet. Call 'fit' before 'summary'.")

        lines = []
        lines.append("ComBat Summary")
        lines.append("=" * 40)
        lines.append(f"Method:          {self.method}")
        lines.append(f"Parametric:      {self.parametric}")
        lines.append(f"Mean only:       {self.mean_only}")
        lines.append(f"Reference batch: {self.reference_batch or 'None'}")

        batch_levels = self._model._batch_levels
        n_per_batch = self._model._n_per_batch
        lines.append(f"Number of batches: {len(batch_levels)}")
        lines.append("Samples per batch:")
        for lvl in batch_levels:
            lines.append(f"  {lvl}: {n_per_batch[str(lvl)]}")

        n_features = len(self._model._grand_mean)
        lines.append(f"Number of features: {n_features}")

        lines.append("")
        lines.append("Top 5 features by batch effect (combined):")
        importance = self.feature_batch_importance()
        top5 = importance.head(5)
        for feat, row in top5.iterrows():
            lines.append(f"  {feat}: {row['combined']:.4f}")

        return "\n".join(lines)

"""Scikit-learn compatible ComBat wrapper."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .core import ArrayLike, ComBatModel
from .metrics import ComBatMetricsMixin
from .visualization import ComBatVisualizationMixin


class ComBat(ComBatMetricsMixin, ComBatVisualizationMixin, BaseEstimator, TransformerMixin):  # type: ignore[misc]
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

    _NEAR_ZERO_VAR_THRESH: float = 1e-10
    _IMBALANCE_RATIO_THRESH: float = 20.0

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
            nan_count = int(batch_ser.isna().sum())
            if nan_count:
                raise ValueError(
                    f"batch contains {nan_count} NaN value(s). "
                    f"All batch labels must be non-null. Check your data for missing entries."
                )

            if self.discrete_covariates is not None:
                disc_df = (
                    pd.DataFrame(self.discrete_covariates)
                    if not isinstance(self.discrete_covariates, pd.Series | pd.DataFrame)
                    else self.discrete_covariates
                )
                nan_count = int(
                    disc_df.isna().sum().sum()
                    if isinstance(disc_df, pd.DataFrame)
                    else disc_df.isna().sum()
                )
                if nan_count:
                    raise ValueError(
                        f"discrete_covariates contains {nan_count} NaN value(s). "
                        f"All covariate values must be non-null."
                    )

            if self.continuous_covariates is not None:
                cont_vals: ArrayLike = self.continuous_covariates
                if isinstance(cont_vals, pd.Series | pd.DataFrame):
                    cont_vals = cont_vals.values  # type: ignore[assignment]
                check_array(
                    np.atleast_2d(cont_vals) if np.asarray(cont_vals).ndim == 1 else cont_vals,
                    ensure_all_finite=True,
                    dtype="numeric",
                )

    def _check_data_quality(
        self,
        X: pd.DataFrame,
        batch_ser: pd.Series,
        disc: pd.DataFrame | pd.Series | None,
        cont: pd.DataFrame | pd.Series | None,
    ) -> None:
        """Issue warnings for data quality issues that may affect results."""
        # Near-zero variance features
        var = X.var(axis=0)
        near_zero = var[var < self._NEAR_ZERO_VAR_THRESH].index.tolist()
        if near_zero:
            preview = near_zero[:5]
            suffix = f"... ({len(near_zero)} total)" if len(near_zero) > 5 else ""
            warnings.warn(
                f"{len(near_zero)} feature(s) have near-zero variance "
                f"(< {self._NEAR_ZERO_VAR_THRESH}): {preview}{suffix}. "
                f"ComBat standardization divides by sqrt(pooled variance), which may "
                f"amplify noise in these features. Consider removing them.",
                UserWarning,
                stacklevel=3,
            )

        # Highly imbalanced batches
        counts = batch_ser.value_counts()
        ratio = counts.max() / counts.min()
        if ratio > self._IMBALANCE_RATIO_THRESH:
            warnings.warn(
                f"Batch sizes are highly imbalanced (ratio {ratio:.1f}:1). "
                f"Largest: '{counts.idxmax()}' ({counts.max()} samples), "
                f"smallest: '{counts.idxmin()}' ({counts.min()} samples). "
                f"Empirical Bayes estimates may be unreliable for small batches.",
                UserWarning,
                stacklevel=3,
            )

        # Covariate collinearity with batch (fortin/chen only)
        if self.method.lower() in {"fortin", "chen"} and (disc is not None or cont is not None):
            batch_dummies = pd.get_dummies(batch_ser, drop_first=True).astype(float)
            cov_parts: list[pd.DataFrame] = []
            if disc is not None:
                d = disc if isinstance(disc, pd.DataFrame) else disc.to_frame()
                cov_parts.append(
                    pd.get_dummies(d.astype("category"), drop_first=True).astype(float)
                )
            if cont is not None:
                c = cont if isinstance(cont, pd.DataFrame) else cont.to_frame()
                cov_parts.append(c.astype(float))
            cov_df = pd.concat(cov_parts, axis=1)
            design = pd.concat([batch_dummies, cov_df], axis=1)
            rank = la.matrix_rank(design.values)
            if rank < design.shape[1]:
                warnings.warn(
                    f"Design matrix is rank-deficient (rank={rank}, "
                    f"columns={design.shape[1]}). One or more covariates may be "
                    f"perfectly collinear with batch indicators, which can lead to "
                    f"unstable parameter estimates. Check whether any covariate "
                    f"perfectly predicts batch membership.",
                    UserWarning,
                    stacklevel=3,
                )

    @staticmethod
    def _batch_variance_explained(X: npt.NDArray[Any], batch_labels: npt.NDArray[Any]) -> float:
        """Fraction of total variance explained by batch (mean across features)."""
        grand_mean = X.mean(axis=0)
        ss_total = float(((X - grand_mean) ** 2).sum())
        if ss_total == 0:
            return 0.0
        ss_between = 0.0
        for lvl in np.unique(batch_labels):
            mask = batch_labels == lvl
            batch_mean = X[mask].mean(axis=0)
            ss_between += float(mask.sum()) * float(((batch_mean - grand_mean) ** 2).sum())
        return ss_between / ss_total

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
            X_df = X
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object
            )
            X_df = pd.DataFrame(X)

        self._model = ComBatModel(
            method=self.method,  # type: ignore[arg-type]
            parametric=self.parametric,
            mean_only=self.mean_only,
            reference_batch=self.reference_batch,
            eps=self.eps,
            covbat_cov_thresh=self.covbat_cov_thresh,
        )

        batch_vec = self._subset(self.batch, idx)
        disc = self._subset(self.discrete_covariates, idx)
        cont = self._subset(self.continuous_covariates, idx)

        self._check_data_quality(X_df, batch_vec, disc, cont)  # type: ignore[arg-type]

        self._model.fit(
            X,
            batch=batch_vec,  # type: ignore[arg-type]
            discrete_covariates=disc,
            continuous_covariates=cont,
        )
        self._fitted_batch = batch_vec

        batch_arr = np.asarray(batch_vec)
        self._batch_var_before_ = self._batch_variance_explained(X_df.values, batch_arr)

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
        X_transformed = self._model.transform(
            X,
            batch=batch_vec,  # type: ignore[arg-type]
            discrete_covariates=disc,
            continuous_covariates=cont,
        )

        batch_arr = np.asarray(batch_vec)
        self._batch_var_after_ = self._batch_variance_explained(X_transformed.values, batch_arr)

        return X_transformed

    @staticmethod
    def _subset(obj: ArrayLike | None, idx: pd.Index) -> pd.DataFrame | pd.Series | None:
        """Subset array-like object by index."""
        if obj is None:
            return None
        if isinstance(obj, pd.Series | pd.DataFrame):
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

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> npt.NDArray[Any]:
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

        lines: list[str] = []
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

        # Diagnostics table
        lines.append("")
        lines.append("Diagnostics")
        lines.append("=" * 40)
        col_w = 34
        lines.append(f"{'Metric':<{col_w}}Value")
        lines.append(f"{'------':<{col_w}}-----")

        if hasattr(self, "_batch_var_before_"):
            lines.append(f"{'Batch var. explained (before)':<{col_w}}{self._batch_var_before_:.1%}")
        if hasattr(self, "_batch_var_after_"):
            lines.append(f"{'Batch var. explained (after)':<{col_w}}{self._batch_var_after_:.1%}")

        design_cond = getattr(self._model, "_design_cond", None)
        if design_cond is not None:
            lines.append(f"{'Design matrix condition number':<{col_w}}{design_cond:.1f}")

        conv_info = getattr(self._model, "_convergence_info", [])
        if conv_info:
            eb_type = "parametric" if self.parametric else "non-parametric"
            lines.append(f"EB convergence ({eb_type}):")
            for info in conv_info:
                batch_name = info["batch"]
                if info["converged"]:
                    status = f"converged ({info['iterations']} iter)"
                else:
                    max_change = max(info["final_gamma_change"], info["final_delta_change"])
                    status = f"NOT CONVERGED ({info['iterations']} iter, \u0394={max_change:.2e})"
                lines.append(f"  {batch_name!s:<{col_w - 2}}{status}")

        return "\n".join(lines)

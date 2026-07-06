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

from ._utils import _subset
from .core import ArrayLike, ComBatModel, _resolve_method


class ComBat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Pipeline-friendly wrapper around `ComBatModel`.

    Stores batch (and optional covariates) passed at construction and
    appropriately uses them for separate `fit` and `transform`.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample.
    discrete_covariates : array-like, optional
        Categorical covariates to protect (Fortin/Chen/Longitudinal only).
    continuous_covariates : array-like, optional
        Continuous covariates to protect (Fortin/Chen/Longitudinal only).
    subject_id : array-like, optional
        Subject/individual labels for the random intercept. Required for
        ``method='longitudinal'``, ignored otherwise.
    time_covariate : array-like, optional
        Continuous time variable for repeated measures (Longitudinal only).
    method : {'johnson', 'fortin', 'chen', 'longitudinal', 'gam', \
'covbat_gam'}, default='johnson'
        ComBat variant to use. 'gam'/'covbat_gam' model the continuous
        covariates in ``smooth_terms`` nonlinearly with B-splines (ComBat-GAM,
        Pomponio et al. 2020). Literature aliases are also accepted:
        'classic_combat' (johnson), 'neurocombat' (fortin), 'covbat' (chen),
        'longcombat' (longitudinal), 'combat_gam' (gam).
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
    smooth_terms : list of str or int, optional
        Continuous covariates to model nonlinearly (gam/covbat_gam only).
        Default (None) smooths every continuous covariate.
    spline_df : int, default=10
        B-spline degrees of freedom per smooth term.
    spline_degree : int, default=3
        B-spline degree (3 = cubic).
    smooth_term_bounds : tuple of (float, float) or dict, optional
        Boundary knots for the splines; a single ``(lo, hi)`` for all terms or a
        ``{term: (lo, hi)}`` dict. Default uses each term's training min/max.
    """

    _NEAR_ZERO_VAR_THRESH: float = 1e-10
    _IMBALANCE_RATIO_THRESH: float = 20.0

    def __init__(
        self,
        batch: ArrayLike,
        *,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        subject_id: ArrayLike | None = None,
        time_covariate: ArrayLike | None = None,
        method: str = "johnson",
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: str | None = None,
        eps: float = 1e-8,
        covbat_cov_thresh: float | int = 0.9,
        smooth_terms: list[str | int] | None = None,
        spline_df: int = 10,
        spline_degree: int = 3,
        smooth_term_bounds: tuple[float, float] | dict[Any, tuple[float, float]] | None = None,
    ) -> None:
        self.batch = batch
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.subject_id = subject_id
        self.time_covariate = time_covariate
        self.method = method
        self.parametric = parametric
        self.mean_only = mean_only
        self.reference_batch = reference_batch
        self.eps = eps
        self.covbat_cov_thresh = covbat_cov_thresh
        self.smooth_terms = smooth_terms
        self.spline_df = spline_df
        self.spline_degree = spline_degree
        self.smooth_term_bounds = smooth_term_bounds

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

            if self.subject_id is not None:
                subj_ser = (
                    pd.Series(self.subject_id)
                    if not isinstance(self.subject_id, pd.Series)
                    else self.subject_id
                )
                nan_count = int(subj_ser.isna().sum())
                if nan_count:
                    raise ValueError(
                        f"subject_id contains {nan_count} NaN value(s). "
                        f"All subject labels must be non-null."
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
        time: pd.DataFrame | pd.Series | None = None,
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

        # Covariate collinearity with batch (covariate-aware methods only)
        if _resolve_method(self.method) in {
            "fortin",
            "chen",
            "longitudinal",
            "gam",
            "covbat_gam",
        } and (disc is not None or cont is not None or time is not None):
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
            if time is not None:
                t = time if isinstance(time, pd.DataFrame) else time.to_frame()
                cov_parts.append(t.astype(float))
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
            smooth_terms=self.smooth_terms,
            spline_df=self.spline_df,
            spline_degree=self.spline_degree,
            smooth_term_bounds=self.smooth_term_bounds,
        )

        batch_vec = _subset(self.batch, idx)
        disc = _subset(self.discrete_covariates, idx)
        cont = _subset(self.continuous_covariates, idx)
        subj = _subset(self.subject_id, idx)
        time = _subset(self.time_covariate, idx)

        self._check_data_quality(X_df, batch_vec, disc, cont, time)  # type: ignore[arg-type]

        self._model.fit(
            X,
            batch=batch_vec,  # type: ignore[arg-type]
            discrete_covariates=disc,
            continuous_covariates=cont,
            subject_id=subj,
            time_covariate=time,
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
        batch_vec = _subset(self.batch, idx)
        disc = _subset(self.discrete_covariates, idx)
        cont = _subset(self.continuous_covariates, idx)
        subj = _subset(self.subject_id, idx)
        time = _subset(self.time_covariate, idx)
        X_transformed = self._model.transform(
            X,
            batch=batch_vec,  # type: ignore[arg-type]
            discrete_covariates=disc,
            continuous_covariates=cont,
            subject_id=subj,
            time_covariate=time,
        )

        batch_arr = np.asarray(batch_vec)
        self._batch_var_after_ = self._batch_variance_explained(X_transformed.values, batch_arr)

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

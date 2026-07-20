"""Transductive ComBat harmonizers (``fit_transform``-only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from .._utils import _check_positional_alignment, _subset
from ..core import ArrayLike, ComBatModel, _resolve_method


class TransductiveComBat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Whole-cohort ComBat harmonizers that do not meet the inductive contract.

    Some ComBat variants only pay off **in-sample**: their benefit is tied to the
    samples present at fit time, so there is no leakage-free way to freeze them on
    a training split and apply them to held-out data. ``TransductiveComBat``
    exposes these as a single ``fit_transform`` step over a complete cohort;
    calling :meth:`transform` on separate held-out data raises, and it is
    deliberately not meant for a scikit-learn ``Pipeline``.

    Currently ``method='longitudinal'`` (Longitudinal ComBat, *Beer et al.* 2020)
    is available.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample.
    method : {'longitudinal'}, default='longitudinal'
        Transductive engine to use. The alias ``'longcombat'`` is also accepted.
    discrete_covariates : array-like, optional
        Categorical covariates to protect.
    continuous_covariates : array-like, optional
        Continuous covariates to protect.
    subject_id : array-like, optional
        Subject/individual labels for the random intercept. Required for
        ``method='longitudinal'``.
    time_covariate : array-like, optional
        Continuous time variable for repeated measures (Longitudinal only).
    parametric : bool, default=True
        Use parametric empirical Bayes.
    mean_only : bool, default=False
        Adjust only the mean (ignore variance).
    reference_batch : str, optional
        Batch level to leave unchanged.
    eps : float, default=1e-8
        Numerical jitter for stability.

    Notes
    -----
    "Transductive" here follows scikit-learn's glossary sense: the estimator "is
    designed to model a specific dataset, but not to apply that model to unseen
    data" - i.e. whole-cohort, non-inductive, ``fit_transform``-only. It is *not*
    the strictly supervised Vapnik sense of transduction (predicting labels for
    specific unlabeled points): ComBat is unsupervised and predicts no labels.
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        method: str = "longitudinal",
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        subject_id: ArrayLike | None = None,
        time_covariate: ArrayLike | None = None,
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: str | None = None,
        eps: float = 1e-8,
    ) -> None:
        self.batch = batch
        self.method = method
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.subject_id = subject_id
        self.time_covariate = time_covariate
        self.parametric = parametric
        self.mean_only = mean_only
        self.reference_batch = reference_batch
        self.eps = eps

    _SUPPORTED_METHODS: frozenset[str] = frozenset({"longitudinal"})

    def _check_method(self) -> str:
        method = _resolve_method(self.method)
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method={self.method!r} is not available in TransductiveComBat. Currently "
                f"supported: {sorted(self._SUPPORTED_METHODS)}. The 'seq' (ComBat-seq) and "
                f"'met' (ComBat-met) engines are planned for a future release."
            )
        return method

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> TransductiveComBat:
        """Fit the underlying whole-cohort model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The complete cohort to harmonize.
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        self : TransductiveComBat
            Fitted estimator.
        """
        self._check_method()
        check_array(X, ensure_all_finite=True, dtype="numeric")
        _check_positional_alignment(
            X,
            [
                ("batch", self.batch),
                ("discrete_covariates", self.discrete_covariates),
                ("continuous_covariates", self.continuous_covariates),
                ("subject_id", self.subject_id),
                ("time_covariate", self.time_covariate),
            ],
        )
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object
            )

        self._model = ComBatModel(
            method=self.method,  # type: ignore[arg-type]
            parametric=self.parametric,
            mean_only=self.mean_only,
            reference_batch=self.reference_batch,
            eps=self.eps,
        )
        self._model.fit(
            X,
            batch=_subset(self.batch, idx),  # type: ignore[arg-type]
            discrete_covariates=_subset(self.discrete_covariates, idx),
            continuous_covariates=_subset(self.continuous_covariates, idx),
            subject_id=_subset(self.subject_id, idx),
            time_covariate=_subset(self.time_covariate, idx),
        )
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> pd.DataFrame:
        """Fit and harmonize the whole cohort in a single pass.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The complete cohort to harmonize.
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        X_transformed : pd.DataFrame
            Batch-corrected data for the whole cohort.
        """
        self.fit(X)
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        return self._model.transform(
            X,
            batch=_subset(self.batch, idx),  # type: ignore[arg-type]
            discrete_covariates=_subset(self.discrete_covariates, idx),
            continuous_covariates=_subset(self.continuous_covariates, idx),
            subject_id=_subset(self.subject_id, idx),
            time_covariate=_subset(self.time_covariate, idx),
        )

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Not supported: ``TransductiveComBat`` is ``fit_transform``-only.

        Raises
        ------
        NotImplementedError
            Always. The correction is in-sample, so it cannot be frozen at fit and
            applied to separate held-out data.
        """
        raise NotImplementedError(
            "TransductiveComBat is fit_transform-only: its correction is realized in-sample, "
            "so it cannot be applied to separate held-out data. Call fit_transform(X) on the "
            "complete cohort instead. For cross-validation-safe harmonization of new samples, "
            "use the inductive combatlearn.ComBat."
        )

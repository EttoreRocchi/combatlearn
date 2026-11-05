"""Scikit-learn transformer for the non-Gaussian GLM ComBat engines."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._utils import (
    _check_positional_alignment,
    _subset,
    _warn_if_covariate_determines_target,
)
from ..core import ArrayLike
from ._model import GLMComBatModel


class GLMComBat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Inductive ComBat for count and proportion data (ComBat-seq, ComBat-met).

    Both engines fit a parametric generalized linear model - negative binomial
    for counts, beta regression for proportions - and then map every
    observation through the CDF of its own batch's fitted distribution into the
    quantile of a batch-free target distribution. That map is **strictly per
    observation**: the value written for a sample depends only on that sample's
    own measurements, batch label, covariates and (``'seq'``) library size.
    Everything else is frozen at ``fit``, so ``transform`` applies to held-out
    data and the estimator is safe inside a cross-validation
    :class:`~sklearn.pipeline.Pipeline`.

    ``X`` must hold non-negative integer counts for ``'seq'`` or values in
    ``[0, 1]`` for ``'met'``. Unlike :class:`~combatlearn.ComBat`, there is no
    ``parametric`` empirical-Bayes switch.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample.
    method : {'seq', 'met'}, default='seq'
        * ``'seq'`` - ComBat-seq (*Zhang, Parmigiani & Johnson* 2020,
          doi:10.1093/nargab/lqaa078). Negative-binomial GLM for raw RNA-seq
          counts, with a per-sample library-size offset and per-feature,
          per-batch dispersions. Integer in, integer out.
        * ``'met'`` - ComBat-met (*Wang* 2025, doi:10.1093/nargab/lqaf062).
          Logit-link beta regression for DNA-methylation beta-values, with
          per-feature, per-batch precisions. Output stays strictly inside
          ``(0, 1)``.

        The aliases ``'combat_seq'`` and ``'combat_met'`` are also accepted.
    discrete_covariates : array-like, optional
        Categorical covariates whose effect is preserved.
    continuous_covariates : array-like, optional
        Continuous covariates whose effect is preserved.
    reference_batch : str, optional
        Batch level to use as the correction target; its samples are returned
        unchanged. **Recommended under cross-validation**, because it pins the
        target mean to one batch instead of to a fold-dependent cohort average.
        See the notes for what it does and does not pin in each engine.
    min_samples_per_batch : int, default=2
        Smallest batch accepted at ``fit``. A batch of one sample has no
        within-batch variance, so its dispersion is unidentifiable.
    pseudo_beta : float, default=1e-4
        ``'met'`` only: the value substituted for observations at exactly 0 or
        1, which fall outside the support of the beta distribution.

    Attributes
    ----------
    feature_names_in_ : ndarray of str
        Feature names seen at fit.
    adjustable_features_ : ndarray of bool
        Per-feature mask of the features the model actually corrects. Features
        outside it are passed through unchanged at every transform.

    Notes
    -----
    **Frozen state.** ``fit`` stores the per-feature background level, the
    per-batch effects, the covariate coefficients, the per-feature/per-batch
    dispersions (or precisions), the batch-free target, and the pass-through
    feature mask. Nothing about the training samples themselves is retained.

    **Feature mask.** ComBat-seq cannot model a feature that is entirely zero
    within some batch, and ComBat-met cannot model one the design reproduces
    exactly; those are frozen as pass-through at fit. Which features that
    catches depends on the training fold, so under cross-validation a feature
    may be corrected in one fold and passed through in another. Filtering
    low-count features once, before cross-validation, avoids this.

    **What ``reference_batch`` pins.** The target mean is size-weighted in both
    engines, so it drifts with batch proportions across folds;
    ``reference_batch`` pins it. For ``'met'`` the target precision is
    size-weighted too and is likewise pinned. For ``'seq'`` the target
    dispersion is an *unweighted* average across batches, which does not drift,
    so it is left unpinned as the lower-variance choice. Both behaviours match
    their reference implementation.

    **Cost.** Both engines are orders of magnitude slower than the Gaussian
    ``ComBat``: the dispersion (or precision) is maximized over an 11-point
    grid per feature and per batch, each grid point a full GLM refit. On one
    modern core with three batches, ``fit_transform`` takes roughly 2.6 s for
    ``'seq'`` at 120 x 5 000 and 19 s at 200 x 20 000; ``'met'`` takes about
    22 s at 120 x 5 000 and 113 s at 120 x 20 000. Cost is close to linear in
    the feature count, so filtering uninformative features first is the most
    effective saving. ``transform`` is much cheaper than ``fit`` (for ``'met'``
    a pure quantile map), so per-fold cost under cross-validation is dominated
    by the fit.

    **Validation status.** The whole-cohort ``fit_transform`` path is verified
    against InMoose ``pycombat_seq`` (``'seq'``) and the ComBat-met R package
    (``'met'``). The inductive ``transform`` has no external reference; it is
    validated by its defining property, that a sample's output does not depend
    on which other samples accompany it.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from combatlearn import GLMComBat
    >>> rng = np.random.default_rng(0)
    >>> counts = pd.DataFrame(rng.poisson(50, size=(20, 8)))
    >>> batch = pd.Series(["a"] * 10 + ["b"] * 10, index=counts.index)
    >>> corrected = GLMComBat(batch=batch, method="seq").fit_transform(counts)
    >>> corrected.shape
    (20, 8)

    See Also
    --------
    combatlearn.ComBat : Gaussian ComBat for continuous data.
    combatlearn.transductive.TransductiveComBat : whole-cohort seq/met variants
        that additionally protect a condition-of-interest term.
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        method: str = "seq",
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        reference_batch: str | None = None,
        min_samples_per_batch: int = 2,
        pseudo_beta: float = 1e-4,
    ) -> None:
        self.batch = batch
        self.method = method
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.reference_batch = reference_batch
        self.min_samples_per_batch = min_samples_per_batch
        self.pseudo_beta = pseudo_beta

    def _alignment_items(self) -> list[tuple[str, ArrayLike | None]]:
        return [
            ("batch", self.batch),
            ("discrete_covariates", self.discrete_covariates),
            ("continuous_covariates", self.continuous_covariates),
        ]

    @staticmethod
    def _frame(X: ArrayLike) -> pd.DataFrame:
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> GLMComBat:
        """Estimate the GLM and freeze every parameter the transform needs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Counts (``method='seq'``) or beta-values (``method='met'``).
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        self : GLMComBat
            Fitted estimator.
        """
        _check_positional_alignment(X, self._alignment_items())
        X_df = self._frame(X)
        idx = X_df.index
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        _warn_if_covariate_determines_target(
            y, _subset(self.discrete_covariates, idx), estimator=type(self).__name__
        )

        self._model = GLMComBatModel(
            method=self.method,  # type: ignore[arg-type]
            reference_batch=self.reference_batch,
            min_samples_per_batch=self.min_samples_per_batch,
            pseudo_beta=self.pseudo_beta,
        )
        self._model.fit(
            X_df,
            batch=_subset(self.batch, idx),  # type: ignore[arg-type]
            discrete_covariates=_subset(self.discrete_covariates, idx),
            continuous_covariates=_subset(self.continuous_covariates, idx),
        )
        self.adjustable_features_ = self._model._adjustable
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Harmonize data using the parameters frozen at fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to harmonize. Every batch label must have been seen at fit.

        Returns
        -------
        X_transformed : pd.DataFrame
            Batch-corrected data, same shape and index as ``X``.
        """
        check_is_fitted(self, "_model")
        _check_positional_alignment(X, self._alignment_items())
        X_df = self._frame(X)
        idx = X_df.index
        return self._model.transform(
            X_df,
            batch=_subset(self.batch, idx),  # type: ignore[arg-type]
            discrete_covariates=_subset(self.discrete_covariates, idx),
            continuous_covariates=_subset(self.continuous_covariates, idx),
        )

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

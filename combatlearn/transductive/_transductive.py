"""Transductive ComBat harmonizers (``fit_transform``-only)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from .._utils import _check_positional_alignment, _subset
from ..core import ArrayLike, ComBatModel, _resolve_method
from ..glm import GLMComBatModel


class TransductiveComBat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Whole-cohort ComBat harmonizers whose *configuration* cannot be frozen at fit.

    These configurations protect a **condition-of-interest / outcome term** in
    the design. That term is unavailable at transform time (and leakage-prone if
    used), so the correction applies only to a complete cohort: use
    ``fit_transform``. :meth:`transform` raises, and the class is not meant for
    a scikit-learn ``Pipeline``.

    A condition **known at intake** (tissue type, treatment arm, sex) is an
    ordinary covariate - inductive, no leakage - and belongs on the inductive
    estimators instead. Use this class only when the protected term *is* the
    supervised target of the downstream analysis, as in differential expression
    and differential methylation.

    .. deprecated:: 2.4.0
        ``method='longitudinal'`` is deprecated here and will be removed in v3.0.0.
        Use :class:`~combatlearn.LongitudinalComBat` (*Beer et al.* 2020): its
        ``fit_transform`` on a whole cohort is bit-identical to this one, and it
        also supports ``transform``.

    Parameters
    ----------
    batch : array-like of shape (n_samples,)
        Batch labels for each sample.
    method : {'seq', 'met', 'longitudinal'}, default='longitudinal'
        Whole-cohort engine to use. Pass it explicitly: the engines take disjoint
        arguments (``condition`` for ``'seq'``/``'met'``, ``subject_id`` for
        ``'longitudinal'``), and the default goes with ``'longitudinal'`` in v3.0.0.

        * ``'seq'`` - condition-protected ComBat-seq (*Zhang, Parmigiani &
          Johnson* 2020) for RNA-seq counts, i.e. the upstream ``full_mod=TRUE``
          configuration. Requires ``condition``.
        * ``'met'`` - condition-protected ComBat-met (*Wang* 2025) for
          DNA-methylation beta-values. Requires ``condition``.
        * ``'longitudinal'`` - deprecated, see above. The alias ``'longcombat'``
          is also accepted.
    condition : array-like, optional
        The condition-of-interest / outcome term to protect in the design.
        Required for ``'seq'`` and ``'met'``. It enters the design exactly as a
        covariate does, so the correction is bit-identical to passing it to
        :class:`~combatlearn.GLMComBat` as a covariate; the difference is that
        :meth:`transform` then refuses to run.
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
        Use parametric empirical Bayes (Longitudinal only).
    mean_only : bool, default=False
        Adjust only the mean, ignoring the variance (Longitudinal only).
    reference_batch : str, optional
        Batch level to leave unchanged.
    min_samples_per_batch : int, default=2
        Smallest batch accepted at fit (``'seq'`` / ``'met'`` only).
    pseudo_beta : float, default=1e-4
        Value substituted for observations at exactly 0 or 1 (``'met'`` only).
    eps : float, default=1e-8
        Numerical jitter for stability (Longitudinal only).

    Notes
    -----
    "Transductive" follows scikit-learn's glossary: the estimator "is designed to
    model a specific dataset, but not to apply that model to unseen data". It is
    not Vapnik's transduction: ComBat is unsupervised and predicts no labels.

    See Also
    --------
    combatlearn.GLMComBat : the inductive ComBat-seq / ComBat-met estimator, for
        when the condition is known at intake and is therefore a covariate.
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        method: str = "longitudinal",
        condition: ArrayLike | None = None,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        subject_id: ArrayLike | None = None,
        time_covariate: ArrayLike | None = None,
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: str | None = None,
        min_samples_per_batch: int = 2,
        pseudo_beta: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        self.batch = batch
        self.method = method
        self.condition = condition
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.subject_id = subject_id
        self.time_covariate = time_covariate
        self.parametric = parametric
        self.mean_only = mean_only
        self.reference_batch = reference_batch
        self.min_samples_per_batch = min_samples_per_batch
        self.pseudo_beta = pseudo_beta
        self.eps = eps

    _SUPPORTED_METHODS: frozenset[str] = frozenset({"seq", "met", "longitudinal"})
    _GLM_METHODS: frozenset[str] = frozenset({"seq", "met"})

    def _check_method(self) -> str:
        method = _resolve_method(self.method)
        if method not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method={self.method!r} is not available in TransductiveComBat. "
                f"Supported: {sorted(self._SUPPORTED_METHODS)}. This class hosts only "
                f"configurations that protect a condition-of-interest term, which is "
                f"unavailable at transform time. For leakage-free harmonization that also "
                f"supports transform on held-out data, use combatlearn.ComBat, "
                f"combatlearn.GLMComBat (counts / proportions), combatlearn.NestedComBat "
                f"(several batch variables) or combatlearn.LongitudinalComBat "
                f"(repeated measures)."
            )
        if method in self._GLM_METHODS:
            if self.condition is None:
                raise ValueError(
                    f"TransductiveComBat(method={self.method!r}) requires `condition`: "
                    f"declaring the outcome term is what makes this configuration "
                    f"whole-cohort. The arithmetic is not what differs - a condition "
                    f"enters the design exactly as a covariate does, so the correction "
                    f"here is bit-identical to GLMComBat with the same variable as a "
                    f"covariate. What this class adds is that transform then refuses to "
                    f"run, so the model cannot be applied to held-out data. With no "
                    f"condition to declare there is nothing left to refuse: use "
                    f"GLMComBat(batch=..., method={method!r}) instead, which additionally "
                    f"supports transform. If your condition is known at intake (tissue "
                    f"type, treatment arm, sex) it is an ordinary covariate: pass it as "
                    f"`discrete_covariates` to GLMComBat."
                )
            if self.subject_id is not None or self.time_covariate is not None:
                warnings.warn(
                    f"`subject_id` and `time_covariate` are only used when "
                    f"method='longitudinal'; they are ignored for method={method!r}.",
                    stacklevel=3,
                )
        elif method == "longitudinal":
            if self.condition is not None:
                warnings.warn(
                    "`condition` is only used by the 'seq' and 'met' engines; it is "
                    "ignored for method='longitudinal'.",
                    stacklevel=3,
                )
            warnings.warn(
                "TransductiveComBat(method='longitudinal') is deprecated and will be removed "
                "in v3.0.0. Longitudinal ComBat is inductive, so it now has its own "
                "transformer: use combatlearn.LongitudinalComBat. This wrapper is redundant - "
                "on a whole cohort every subject is seen at fit, so "
                "LongitudinalComBat.fit_transform returns bit-identical output while also "
                "supporting transform on held-out data.",
                DeprecationWarning,
                stacklevel=3,
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
        method = self._check_method()
        check_array(X, ensure_all_finite=True, dtype="numeric")
        _check_positional_alignment(
            X,
            [
                ("batch", self.batch),
                ("condition", self.condition),
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

        if method in self._GLM_METHODS:
            self._model: GLMComBatModel | ComBatModel = GLMComBatModel(
                method=method,  # type: ignore[arg-type]
                reference_batch=self.reference_batch,
                min_samples_per_batch=self.min_samples_per_batch,
                pseudo_beta=self.pseudo_beta,
            )
            self._model.fit(
                X,
                batch=_subset(self.batch, idx),  # type: ignore[arg-type]
                discrete_covariates=_subset(self.discrete_covariates, idx),
                continuous_covariates=_subset(self.continuous_covariates, idx),
                condition=_subset(self.condition, idx),
            )
            return self

        self._model = ComBatModel(
            method=method,  # type: ignore[arg-type]
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
        if isinstance(self._model, GLMComBatModel):
            return self._model._apply(
                X,
                batch=_subset(self.batch, idx),  # type: ignore[arg-type]
                discrete_covariates=_subset(self.discrete_covariates, idx),
                continuous_covariates=_subset(self.continuous_covariates, idx),
                condition=_subset(self.condition, idx),
            )
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
            Always: the protected term is unavailable at transform time, so the
            model cannot be applied to separate held-out data.
        """
        raise NotImplementedError(
            "TransductiveComBat is fit_transform-only: its configuration protects a term "
            "that is unavailable (and leakage-prone) at transform time, so it cannot be "
            "applied to separate held-out data. Call fit_transform(X) on the complete cohort "
            "instead. For cross-validation-safe harmonization of new samples use the "
            "inductive combatlearn.ComBat, combatlearn.GLMComBat for counts or "
            "proportions, or combatlearn.LongitudinalComBat for repeated-measures designs."
        )

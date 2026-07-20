"""Scikit-learn compatible ComBat wrapper."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ._nested import (
    GMM_VAR_NAME,
    GMMGrouping,
    ModelStep,
    fit_nested_sequence,
    search_order,
    transform_nested_sequence,
    variance_explained_by_batch,
)
from ._utils import _check_positional_alignment, _subset
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

    def _auxiliary_items(self) -> list[tuple[str, ArrayLike | None]]:
        """Construction-time vectors that must align with X, as ``(name, value)``."""
        return [
            ("batch", self.batch),
            ("discrete_covariates", self.discrete_covariates),
            ("continuous_covariates", self.continuous_covariates),
            ("subject_id", self.subject_id),
            ("time_covariate", self.time_covariate),
        ]

    def _validate_inputs(self, X: ArrayLike, *, fitting: bool = False) -> None:
        """Validate X and, during fitting, batch/covariates for NaN/Inf."""
        check_array(X, ensure_all_finite=True, dtype="numeric")
        _check_positional_alignment(X, self._auxiliary_items())

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

    def _check_data_quality(self, X: pd.DataFrame, batch_ser: pd.Series) -> None:
        """Issue warnings for data quality issues that may affect results.

        Covariate collinearity is checked separately in :meth:`_warn_rank_deficient`
        after fitting, from the actual least-squares design (which, for the GAM
        engines, is the spline basis rather than the raw continuous covariate).
        """
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

    def _warn_rank_deficient(self) -> None:
        """Warn if the fitted least-squares design is rank-deficient.

        Uses the rank of the design actually used by the fitted model, so the
        check is exact for the covariate-aware engines and, unlike an
        approximation from the raw continuous covariate, correctly covers the
        spline basis of the ``gam``/``covbat_gam`` engines.
        """
        rank = getattr(self._model, "_design_rank", None)
        ncols = getattr(self._model, "_design_ncols", None)
        if rank is None or ncols is None or rank >= ncols:
            return
        warnings.warn(
            f"Design matrix is rank-deficient (rank={rank}, columns={ncols}). One or "
            f"more covariates may be perfectly collinear with the batch indicators "
            f"(or, for the GAM engines, within the spline basis), which can lead to "
            f"unstable parameter estimates. Check whether any covariate perfectly "
            f"predicts batch membership.",
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
        if _resolve_method(self.method) == "longitudinal":
            warnings.warn(
                "method='longitudinal' on the inductive ComBat transformer is deprecated "
                "and will be removed in v3.0.0. Longitudinal ComBat is a whole-cohort "
                "harmonizer; use combatlearn.transductive.TransductiveComBat("
                "method='longitudinal') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
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

        self._check_data_quality(X_df, batch_vec)  # type: ignore[arg-type]

        self._model.fit(
            X,
            batch=batch_vec,  # type: ignore[arg-type]
            discrete_covariates=disc,
            continuous_covariates=cont,
            subject_id=subj,
            time_covariate=time,
        )
        self._warn_rank_deficient()
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


class NestedComBat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Nested / OPNested / GMM ComBat for multiple batch variables.

    Harmonizes over several batch variables at once (e.g. site, scanner,
    protocol; *Horng et al.* 2022) by applying single-batch ComBat to each one in
    sequence, every step delegating to a :class:`~combatlearn.core.ComBatModel`.
    It adds no new empirical-Bayes math - the chosen order, the optional
    Gaussian-mixture grouping, and every per-step parameter are learned on the
    training data and frozen for transform, so it is inductive and
    cross-validation-safe like :class:`ComBat`.

    Parameters
    ----------
    batch : pd.DataFrame or list of array-like
        The batch variables to harmonize. A DataFrame uses one column per batch
        variable (column names become the variable names); a list/tuple provides
        one array-like per batch variable (named Series keep their name, others
        are named ``'batch0'``, ``'batch1'``, ...).
    discrete_covariates : array-like, optional
        Categorical covariates to protect, preserved across every step.
    continuous_covariates : array-like, optional
        Continuous covariates to protect, preserved across every step. Required
        for the GAM engines.
    method : {'fortin', 'chen', 'gam', 'covbat_gam'}, default='fortin'
        The ComBat engine used for every nested step. Literature aliases
        (``'neurocombat'``, ``'covbat'``, ``'combat_gam'``, ``'covbatgam'``) are
        also accepted. ``'johnson'`` and ``'longitudinal'`` are not supported
        (they do not preserve covariates across the nested steps).
    optimize_order : bool, default=True
        If True, select the harmonization order that minimizes the residual batch
        effect (OPNested); otherwise use the order the batch variables are given.
    order_metric : {'anderson'}, default='anderson'
        Objective for the order search: the number of features with a significant
        residual batch effect by the Anderson-Darling k-sample test, summed over
        all batch variables (lower is better).
    max_exhaustive_vars : int, default=4
        Cap on the exhaustive order search. With ``k <= max_exhaustive_vars``
        batch variables all ``k!`` orderings are tried (each ordering fits ``k``
        ComBat models and scores every feature), so the cost grows factorially; a
        warning reports the number of fits before a large exhaustive search runs.
        Above the cap the search falls back to greedy forward selection. Raise this
        to force the exhaustive search over more variables, at your own cost.
    gmm : {None, 'batch', 'covariate'}, default=None
        Optional Gaussian-mixture grouping (GMM ComBat). ``'batch'`` (``+GMM``)
        feeds the latent grouping in as an extra batch variable (harmonized away);
        ``'covariate'`` (``-GMM``) feeds it in as a protected discrete covariate
        (preserved as signal). ``None`` disables it.
    gmm_min_cluster_frac : float, default=0.25
        Minimum fraction of samples each mixture component must hold for a feature
        to be eligible as the grouping source.
    parametric : bool, default=True
        Use parametric empirical Bayes (passed to every step).
    mean_only : bool, default=False
        Adjust only the mean (passed to every step).
    reference_batch : dict or str, optional
        Reference level per batch variable, as a ``{batch_variable: level}`` dict
        (each nested step leaves its reference level unchanged; variables absent
        from the dict use the grand mean). A bare string is accepted only when
        there is a single batch variable. ``None`` uses the grand mean throughout.
    eps : float, default=1e-8
        Numerical jitter (passed to every step).
    covbat_cov_thresh : float or int, default=0.9
        CovBat variance threshold for PCs (``chen`` / ``covbat_gam`` steps).
    smooth_terms : list of str or int, optional
        Continuous covariates to model nonlinearly (``gam`` / ``covbat_gam``).
    spline_df : int, default=10
        B-spline degrees of freedom per smooth term (GAM engines).
    spline_degree : int, default=3
        B-spline degree (GAM engines).
    smooth_term_bounds : tuple of (float, float) or dict, optional
        Boundary knots for the splines (GAM engines).
    random_state : int or None, default=None
        Seed for the Gaussian-mixture initialization (used only when ``gmm`` is
        set). ``None`` follows the scikit-learn convention (nondeterministic
        grouping); pass an int for a reproducible grouping.

    Attributes
    ----------
    order_ : list of str
        The batch variables in the order they were harmonized.
    used_greedy_ : bool
        Whether the greedy fallback was used instead of the exhaustive search.
    batch_var_before_ : dict of str to float
        Per-variable fraction of variance explained by batch before correction.
    batch_var_after_ : dict of str to float
        Per-variable fraction of variance explained by batch after correction.
    """

    def __init__(
        self,
        batch: pd.DataFrame | Sequence[ArrayLike],
        *,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        method: str = "fortin",
        optimize_order: bool = True,
        order_metric: str = "anderson",
        max_exhaustive_vars: int = 4,
        gmm: str | None = None,
        gmm_min_cluster_frac: float = 0.25,
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: dict[Any, str] | str | None = None,
        eps: float = 1e-8,
        covbat_cov_thresh: float | int = 0.9,
        smooth_terms: list[str | int] | None = None,
        spline_df: int = 10,
        spline_degree: int = 3,
        smooth_term_bounds: tuple[float, float] | dict[Any, tuple[float, float]] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.batch = batch
        self.discrete_covariates = discrete_covariates
        self.continuous_covariates = continuous_covariates
        self.method = method
        self.optimize_order = optimize_order
        self.order_metric = order_metric
        self.max_exhaustive_vars = max_exhaustive_vars
        self.gmm = gmm
        self.gmm_min_cluster_frac = gmm_min_cluster_frac
        self.parametric = parametric
        self.mean_only = mean_only
        self.reference_batch = reference_batch
        self.eps = eps
        self.covbat_cov_thresh = covbat_cov_thresh
        self.smooth_terms = smooth_terms
        self.spline_df = spline_df
        self.spline_degree = spline_degree
        self.smooth_term_bounds = smooth_term_bounds
        self.random_state = random_state

    _SUPPORTED_METHODS: frozenset[str] = frozenset({"fortin", "chen", "gam", "covbat_gam"})

    @staticmethod
    def _as_frame(obj: pd.DataFrame | pd.Series | None) -> pd.DataFrame | None:
        """Normalize a subset covariate object to a DataFrame (or None)."""
        if obj is None:
            return None
        return obj.to_frame() if isinstance(obj, pd.Series) else obj

    def _batch_items(self) -> list[tuple[str, ArrayLike]]:
        """Ordered ``(name, raw values)`` pairs, one per batch variable."""
        batch: Any = self.batch
        if isinstance(batch, pd.DataFrame):
            return [(str(col), batch[col]) for col in batch.columns]
        if isinstance(batch, pd.Series):
            name = str(batch.name) if batch.name is not None else "batch0"
            return [(name, batch)]
        if isinstance(batch, list | tuple):
            items: list[tuple[str, ArrayLike]] = []
            for i, values in enumerate(batch):
                if isinstance(values, pd.Series) and values.name is not None:
                    items.append((str(values.name), values))
                else:
                    items.append((f"batch{i}", values))
            return items
        arr = np.asarray(batch)
        if arr.ndim == 2:
            return [(f"batch{i}", arr[:, i]) for i in range(arr.shape[1])]
        return [("batch0", batch)]

    def _alignment_items(self) -> list[tuple[str, ArrayLike | None]]:
        """Batch variables and covariates that must align with X, as ``(name, value)``."""
        return [
            *self._batch_items(),
            ("discrete_covariates", self.discrete_covariates),
            ("continuous_covariates", self.continuous_covariates),
        ]

    def _resolve_batch_vars(self, idx: pd.Index) -> dict[str, pd.Series]:
        """Subset each batch variable to ``idx`` and validate names/values."""
        items = self._batch_items()
        names = [name for name, _ in items]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Batch variable names must be unique, got {names}. Rename the duplicated "
                f"columns/series."
            )
        batch_vars: dict[str, pd.Series] = {}
        for name, raw in items:
            sub = _subset(raw, idx)
            ser = (
                sub if isinstance(sub, pd.Series) else pd.Series(np.asarray(sub).ravel(), index=idx)
            )
            ser = ser.rename(name)
            if ser.isna().any():
                raise ValueError(
                    f"Batch variable '{name}' contains {int(ser.isna().sum())} NaN value(s). "
                    f"All batch labels must be non-null."
                )
            batch_vars[name] = ser
        return batch_vars

    def _resolve_reference_batch(self, names: Sequence[str]) -> dict[str, str | None]:
        """Resolve ``reference_batch`` to a per-variable ``{name: level or None}``."""
        ref = self.reference_batch
        if ref is None:
            return dict.fromkeys(names, None)
        if isinstance(ref, dict):
            unknown = set(ref) - set(names)
            if unknown:
                raise ValueError(
                    f"reference_batch keys {sorted(map(str, unknown))} are not batch "
                    f"variables. Available: {list(names)}."
                )
            return {name: ref.get(name) for name in names}
        if len(names) != 1:
            raise ValueError(
                f"reference_batch={ref!r} is a single level but there are {len(names)} batch "
                f"variables ({list(names)}). Pass a {{batch_variable: level}} dict to set a "
                f"reference level per variable."
            )
        return {names[0]: ref}

    def _make_model(self, name: str) -> ComBatModel:
        """Build a fresh per-step ``ComBatModel`` with the shared parameters."""
        return ComBatModel(
            method=self.method,  # type: ignore[arg-type]
            parametric=self.parametric,
            mean_only=self.mean_only,
            reference_batch=self._reference_by_var[name],
            eps=self.eps,
            covbat_cov_thresh=self.covbat_cov_thresh,
            smooth_terms=self.smooth_terms,
            spline_df=self.spline_df,
            spline_degree=self.spline_degree,
            smooth_term_bounds=self.smooth_term_bounds,
        )

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> NestedComBat:
        """Fit the nested model: select an order and fit one step per batch variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit.
        y : None
            Ignored. Present for API compatibility.

        Returns
        -------
        self : NestedComBat
            Fitted estimator.
        """
        check_array(X, ensure_all_finite=True, dtype="numeric")
        _check_positional_alignment(X, self._alignment_items())
        if _resolve_method(self.method) not in self._SUPPORTED_METHODS:
            raise ValueError(
                f"method={self.method!r} is not supported by NestedComBat. Expected one of "
                f"{sorted(self._SUPPORTED_METHODS)} (or a matching alias); 'johnson' and "
                f"'longitudinal' cannot preserve covariates across the nested steps."
            )
        if self.order_metric != "anderson":
            raise ValueError(
                f"order_metric={self.order_metric!r} is not recognized. The only supported "
                f"metric is 'anderson' (Anderson-Darling k-sample feature count)."
            )
        if self.gmm not in (None, "batch", "covariate"):
            raise ValueError(
                f"gmm={self.gmm!r} is not recognized. Expected None, 'batch' (+GMM, extra "
                f"batch variable) or 'covariate' (-GMM, protected covariate)."
            )

        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            X_df = X
        else:
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(np.asarray(X).shape[1])], dtype=object
            )
            X_df = pd.DataFrame(X)

        batch_vars = self._resolve_batch_vars(idx)
        disc_df = self._as_frame(_subset(self.discrete_covariates, idx))
        cont_df = self._as_frame(_subset(self.continuous_covariates, idx))

        self._gmm_grouping = None
        if self.gmm is not None:
            grouping = GMMGrouping.fit(
                X_df,
                min_cluster_frac=self.gmm_min_cluster_frac,
                random_state=self.random_state,
            )
            if grouping is None:
                warnings.warn(
                    "No feature produced a balanced two-component Gaussian mixture "
                    f"(each cluster > {self.gmm_min_cluster_frac:.0%} of samples), so the GMM "
                    "grouping is skipped and NestedComBat proceeds without it.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self._gmm_grouping = grouping
                labels = grouping.assign(X_df)
                if self.gmm == "batch":
                    if GMM_VAR_NAME in batch_vars:
                        raise ValueError(
                            f"A batch variable is already named {GMM_VAR_NAME!r}, which clashes "
                            f"with the GMM grouping. Rename it before using gmm='batch'."
                        )
                    batch_vars[GMM_VAR_NAME] = labels
                else:  # 'covariate'
                    gmm_cov = labels.to_frame(GMM_VAR_NAME)
                    if disc_df is not None and GMM_VAR_NAME in disc_df.columns:
                        raise ValueError(
                            f"A discrete covariate is already named {GMM_VAR_NAME!r}, which "
                            f"clashes with the GMM grouping. Rename it before using "
                            f"gmm='covariate'."
                        )
                    disc_df = gmm_cov if disc_df is None else pd.concat([disc_df, gmm_cov], axis=1)

        names = list(batch_vars.keys())
        self._reference_by_var = self._resolve_reference_batch(names)

        if self.optimize_order and len(names) > 1:
            order, models, used_greedy = search_order(
                names,
                X_df,
                batch_vars,
                disc_df,
                cont_df,
                self._make_model,
                max_exhaustive_vars=self.max_exhaustive_vars,
            )
        else:
            models, _ = fit_nested_sequence(
                names,
                X_df,
                batch_vars,
                disc_df,
                cont_df,
                self._make_model,
            )
            order, used_greedy = names, False

        self._models: list[ModelStep] = models
        self.order_ = order
        self.used_greedy_ = used_greedy

        X_harmonized = transform_nested_sequence(models, X_df, batch_vars, disc_df, cont_df)
        self.batch_var_before_ = {
            name: variance_explained_by_batch(X_df, batch_vars[name]) for name in names
        }
        self.batch_var_after_ = {
            name: variance_explained_by_batch(X_harmonized, batch_vars[name]) for name in names
        }
        return self

    def transform(self, X: ArrayLike) -> pd.DataFrame:
        """Transform new data by replaying the fitted nested sequence.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        X_transformed : pd.DataFrame
            Batch-corrected data.
        """
        check_is_fitted(self, "_models")
        check_array(X, ensure_all_finite=True, dtype="numeric")
        _check_positional_alignment(X, self._alignment_items())
        idx = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        batch_vars = self._resolve_batch_vars(idx)
        disc_df = self._as_frame(_subset(self.discrete_covariates, idx))
        cont_df = self._as_frame(_subset(self.continuous_covariates, idx))

        if self._gmm_grouping is not None:
            labels = self._gmm_grouping.assign(X_df)
            if self.gmm == "batch":
                batch_vars[GMM_VAR_NAME] = labels
            else:  # 'covariate'
                gmm_cov = labels.to_frame(GMM_VAR_NAME)
                disc_df = gmm_cov if disc_df is None else pd.concat([disc_df, gmm_cov], axis=1)

        return transform_nested_sequence(self._models, X_df, batch_vars, disc_df, cont_df)

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

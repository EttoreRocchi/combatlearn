"""ComBat algorithm core.

`ComBatModel` implements several variants of the ComBat algorithm:
    * Johnson et al. (2007) vanilla ComBat (method="johnson")
    * Fortin et al. (2018) extension with covariates (method="fortin")
    * Chen et al. (2022) CovBat (method="chen")
    * Beer et al. (2020) Longitudinal ComBat (method="longitudinal")
    * Pomponio et al. (2020) ComBat-GAM, spline covariates (method="gam")
    * CovBat with GAM covariates (method="covbat_gam")
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.decomposition import PCA

from ._mixed import fit_random_intercept
from ._utils import _code_levels, _fit_levels

ArrayLike: TypeAlias = pd.DataFrame | pd.Series | npt.NDArray[Any]
FloatArray: TypeAlias = npt.NDArray[np.float64]

_METHOD_ALIASES: dict[str, str] = {
    "classiccombat": "johnson",
    "neurocombat": "fortin",
    "covbat": "chen",
    "longcombat": "longitudinal",
    "combatgam": "gam",
    "covbatgam": "covbat_gam",
    "combatseq": "seq",
    "combatmet": "met",
}

_GAM_METHODS: frozenset[str] = frozenset({"gam", "covbat_gam"})
_EPRIOR_BLOCK_ELEMENTS = 1 << 22
_PC_RELATIVE_VARIANCE_FLOOR = 1e-10


def _resolve_method(method: str) -> str:
    """Resolve a method name or literature alias to its canonical (author) name.

    Case- and separator-insensitive, e.g. ``"CovBat-GAM"`` -> ``"covbat_gam"``. The
    GLM aliases (``"combat_seq"`` -> ``"seq"``, ``"combat_met"`` -> ``"met"``) resolve
    here too; each estimator validates the result against the methods it supports.
    """
    raw = method.lower()
    stripped = raw.replace("-", "").replace("_", "").replace(" ", "")
    return _METHOD_ALIASES.get(stripped, raw)


class ComBatModel:
    """ComBat algorithm.

    Parameters
    ----------
    method : {'johnson', 'fortin', 'chen', 'longitudinal', 'gam', \
'covbat_gam'}, default='johnson'
        * 'johnson' - classic ComBat.
        * 'fortin' - covariate-aware ComBat.
        * 'chen' - CovBat, PCA-based ComBat.
        * 'longitudinal' - Longitudinal ComBat; requires a per-sample
          ``subject_id`` for the random intercept.
        * 'gam' - ComBat-GAM (Pomponio et al. 2020); like 'fortin' but the
          continuous covariates in ``smooth_terms`` are modeled with B-spline
          bases instead of linearly.
        * 'covbat_gam' - CovBat with the same spline covariate modeling.

        Literature aliases are also accepted (case- and separator-insensitive):
        ``'classic_combat'`` (johnson), ``'neurocombat'`` (fortin),
        ``'covbat'`` (chen), ``'longcombat'`` (longitudinal),
        ``'combat_gam'`` (gam).
    parametric : bool, default=True
        Use the parametric empirical Bayes variant.
    mean_only : bool, default=False
        If True, only the mean is adjusted (`gamma_star`),
        ignoring the variance (`delta_star`). For 'chen'/'covbat_gam' this applies to
        the ComBat step only; PC-score variances are still harmonized.
    reference_batch : str, optional
        If specified, the batch level to use as reference.
    covbat_cov_thresh : float or int, default=0.9
        CovBat: cumulative variance threshold (0, 1] to retain PCs, or
        integer >= 1 specifying the number of components directly.
    smooth_terms : list of str or int, optional
        Continuous-covariate columns to model nonlinearly with B-splines
        (``'gam'``/``'covbat_gam'`` only). Column names, or integer positions
        into ``continuous_covariates`` when it is an unnamed array. Default
        (``None``) smooths every continuous covariate.
    spline_df : int, default=10
        Degrees of freedom (number of B-spline basis functions) per smooth
        term, before the identifiability column is dropped. Must be
        ``>= spline_degree + 1``.
    spline_degree : int, default=3
        B-spline degree (3 = cubic). Must be ``>= 1``.
    smooth_term_bounds : tuple of (float, float) or dict, optional
        Boundary knots for the spline basis. A single ``(lo, hi)`` tuple is
        applied to every smooth term; a ``{term: (lo, hi)}`` dict sets them per
        term. Bounds should contain the full data range (training and any
        held-out data). Default uses each term's training min/max.
    eps : float, default=1e-8
        Numerical jitter to avoid division-by-zero.
    """

    def __init__(
        self,
        *,
        method: Literal[
            "johnson", "fortin", "chen", "longitudinal", "gam", "covbat_gam"
        ] = "johnson",
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
        self.method: str = method
        self.parametric: bool = parametric
        self.mean_only: bool = bool(mean_only)
        self.reference_batch: str | None = reference_batch
        self.eps: float = float(eps)
        self.covbat_cov_thresh: float | int = covbat_cov_thresh
        self.smooth_terms: list[str | int] | None = smooth_terms
        self.spline_df: int = int(spline_df)
        self.spline_degree: int = int(spline_degree)
        self.smooth_term_bounds: tuple[float, float] | dict[Any, tuple[float, float]] | None = (
            smooth_term_bounds
        )

        self._batch_levels: pd.Index
        self._grand_mean: pd.Series
        self._pooled_var: pd.Series
        self._gamma_star: FloatArray
        self._delta_star: FloatArray
        self._n_per_batch: dict[str, int]
        self._reference_batch_idx: int | None
        self._beta_hat_nonbatch: FloatArray
        self._nonbatch_columns: list[str]
        self._n_batch: int
        self._p_design: int
        self._covbat_pca: PCA
        self._covbat_n_pc: int
        self._pc_gamma: FloatArray
        self._pc_delta: FloatArray
        self._pc_score_mean: FloatArray
        self._pc_score_var: FloatArray
        self._re_subject_levels: pd.Index
        self._re_blup: FloatArray
        self._smooth_cols: list[Any]
        self._smooth_knots: dict[Any, FloatArray]
        self._smooth_basis_columns: dict[Any, list[str]]
        self._design_cond: float | None
        self._design_rank: int | None
        self._design_ncols: int | None
        self._discrete_levels: dict[Any, pd.Index] = {}

        if self.spline_degree < 1:
            raise ValueError(
                f"spline_degree={self.spline_degree!r} is invalid. Expected an int >= 1 "
                f"(3 = cubic, the ComBat-GAM default)."
            )
        if self.spline_df < self.spline_degree + 1:
            raise ValueError(
                f"spline_df={self.spline_df!r} is too small for spline_degree="
                f"{self.spline_degree!r}. Expected spline_df >= spline_degree + 1 "
                f"(>= {self.spline_degree + 1}); the default is 10."
            )

        if isinstance(self.covbat_cov_thresh, float):
            if not (0.0 < self.covbat_cov_thresh <= 1.0):
                raise ValueError(
                    f"covbat_cov_thresh={self.covbat_cov_thresh!r} is out of range. "
                    f"Expected a float in (0, 1] (e.g. 0.9 to retain 90% of variance)."
                )
        elif isinstance(self.covbat_cov_thresh, int):
            if self.covbat_cov_thresh < 1:
                raise ValueError(
                    f"covbat_cov_thresh={self.covbat_cov_thresh!r} is invalid. "
                    f"Expected an int >= 1 (the number of principal components to retain)."
                )
        else:
            raise TypeError(
                f"covbat_cov_thresh must be float or int, "
                f"got {type(self.covbat_cov_thresh).__name__} ({self.covbat_cov_thresh!r}). "
                f"Use a float in (0, 1] for cumulative variance or an int >= 1 for "
                f"a fixed number of PCs."
            )

    @staticmethod
    def _as_series(arr: ArrayLike, index: pd.Index, name: str) -> pd.Series:
        """Convert array-like to categorical Series with validation."""
        ser = arr.copy() if isinstance(arr, pd.Series) else pd.Series(arr, index=index, name=name)
        if not ser.index.equals(index):
            diff = index.symmetric_difference(ser.index)
            raise ValueError(
                f"`{name}` index does not align with `X`. "
                f"{len(diff)} mismatched entries (first 3): {list(diff[:3])}. "
                f"Ensure both share the same index."
            )
        return ser.astype("category")

    @staticmethod
    def _to_df(arr: ArrayLike | None, index: pd.Index, name: str) -> pd.DataFrame | None:
        """Convert array-like to DataFrame."""
        if arr is None:
            return None
        if isinstance(arr, pd.Series):
            arr = arr.to_frame()
        if not isinstance(arr, pd.DataFrame):
            arr = pd.DataFrame(arr, index=index)
        if not arr.index.equals(index):
            diff = index.symmetric_difference(arr.index)
            raise ValueError(
                f"`{name}` index does not align with `X`. "
                f"{len(diff)} mismatched entries (first 3): {list(diff[:3])}. "
                f"Ensure both share the same index."
            )
        return arr

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        batch: ArrayLike,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        subject_id: ArrayLike | None = None,
        time_covariate: ArrayLike | None = None,
    ) -> ComBatModel:
        """Fit the ComBat model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to fit.
        y : None
            Ignored. Present for API compatibility.
        batch : array-like of shape (n_samples,)
            Batch labels for each sample.
        discrete_covariates : array-like or None, default=None
            Categorical covariates to protect (Fortin/Chen/Longitudinal only).
        continuous_covariates : array-like or None, default=None
            Continuous covariates to protect (Fortin/Chen/Longitudinal only).
        subject_id : array-like or None, default=None
            Subject/individual labels for the random intercept. Required for
            ``method='longitudinal'``, ignored otherwise.
        time_covariate : array-like or None, default=None
            Continuous time variable for repeated measures (Longitudinal only).
            Added to the fixed-effects design alongside ``continuous_covariates``.

        Returns
        -------
        self : ComBatModel
            Fitted model.
        """
        method = _resolve_method(self.method)
        if method in {"seq", "met"}:
            raise ValueError(
                f"method={self.method!r} is a non-Gaussian GLM engine and lives on a "
                f"separate estimator, because the admissible domain of X differs "
                f"(counts for 'seq', proportions for 'met'). Use "
                f"combatlearn.GLMComBat(batch=..., method={method!r}) instead."
            )
        if method not in {"johnson", "fortin", "chen", "longitudinal", "gam", "covbat_gam"}:
            raise ValueError(
                f"method={self.method!r} is not recognized. Expected an author name "
                f"('johnson', 'fortin', 'chen', 'longitudinal', 'gam', 'covbat_gam') or a "
                f"method-name alias ('classic_combat', 'neurocombat', 'covbat', "
                f"'longcombat', 'combat_gam')."
            )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.shape[1] < 2:
            warnings.warn(
                "Only one feature: empirical-Bayes shrinkage is inactive and ComBat "
                "reduces to per-batch standardization.",
                stacklevel=2,
            )
        idx = X.index
        batch = self._as_series(batch, idx, "batch")

        disc = self._to_df(discrete_covariates, idx, "discrete_covariates")
        cont = self._to_df(continuous_covariates, idx, "continuous_covariates")

        if self.reference_batch is not None and self.reference_batch not in batch.cat.categories:
            raise ValueError(
                f"reference_batch={self.reference_batch!r} not found. "
                f"Available batches: {list(batch.cat.categories)}. "
                f"Check for typos or trailing whitespace in batch labels."
            )

        if method != "longitudinal" and (subject_id is not None or time_covariate is not None):
            warnings.warn(
                "`subject_id` and `time_covariate` are only used when "
                "method='longitudinal'; they are ignored here.",
                stacklevel=2,
            )

        if method == "johnson":
            if disc is not None or cont is not None:
                warnings.warn("Covariates are ignored when using method='johnson'.", stacklevel=2)
            self._fit_johnson(X, batch)
        elif method in {"fortin", "gam"}:
            self._fit_fortin(X, batch, disc, cont)
        elif method in {"chen", "covbat_gam"}:
            self._fit_chen(X, batch, disc, cont)
        elif method == "longitudinal":
            self._fit_longitudinal(X, batch, disc, cont, subject_id, time_covariate)
        return self

    def _fit_johnson(self, X: pd.DataFrame, batch: pd.Series) -> None:
        """Johnson et al. (2007) ComBat."""
        self._batch_levels = batch.cat.categories
        grand_mean = X.mean(axis=0)

        n_samples = len(X)
        resid = X.copy()
        for lvl in self._batch_levels:
            idx = batch == lvl
            resid.loc[idx] = X.loc[idx] - X.loc[idx].mean(axis=0)
        pooled_var = (resid**2).sum(axis=0) / n_samples + self.eps

        Xs = (X - grand_mean) / np.sqrt(pooled_var)

        n_per_batch: dict[str, int] = {}
        gamma_hat: list[npt.NDArray[np.float64]] = []
        delta_hat: list[npt.NDArray[np.float64]] = []

        for lvl in self._batch_levels:
            idx = batch == lvl
            n_b = int(idx.sum())
            if n_b < 2:
                raise ValueError(
                    f"Batch '{lvl}' has only {n_b} sample(s). ComBat requires >= 2 samples "
                    f"per batch. Consider merging small batches or removing them."
                )
            n_per_batch[str(lvl)] = n_b
            xb = Xs.loc[idx]
            gamma_hat.append(xb.mean(axis=0).values)
            delta_hat.append(xb.var(axis=0, ddof=1).values + self.eps)

        gamma_hat_arr = np.vstack(gamma_hat)
        delta_hat_arr = np.vstack(delta_hat)

        b_names = [str(lvl) for lvl in self._batch_levels]
        gamma_star, delta_star = self._shrink_eb(gamma_hat_arr, delta_hat_arr, n_per_batch, b_names)
        gamma_star, delta_star, self._reference_batch_idx = self._apply_reference_shift(
            gamma_star,
            delta_star,
            self._batch_levels,
            gamma_hat=gamma_hat_arr,
            delta_hat=delta_hat_arr,
        )

        self._grand_mean = grand_mean
        self._pooled_var = pooled_var
        self._gamma_star = gamma_star
        self._delta_star = delta_star
        self._n_per_batch = n_per_batch
        self._design_cond = None
        self._design_rank = None
        self._design_ncols = None

    def _is_gam(self) -> bool:
        """Whether the resolved method models smooth terms with B-splines."""
        return _resolve_method(self.method) in _GAM_METHODS

    def _resolve_smooth_columns(self, cont: pd.DataFrame | None) -> list[Any]:
        """Resolve ``smooth_terms`` to concrete column labels in ``cont``.

        Raises if no continuous covariates are available (a GAM method needs at
        least one smooth term) or if a requested term cannot be located.
        """
        if cont is None or cont.shape[1] == 0:
            raise ValueError(
                f"method={self.method!r} (ComBat-GAM) requires at least one continuous "
                f"covariate to model with splines, but `continuous_covariates` is empty. "
                f"Pass continuous covariates, or use method='fortin'/'chen' for a linear fit."
            )
        cols = list(cont.columns)
        if self.smooth_terms is None:
            return cols
        if len(self.smooth_terms) == 0:
            raise ValueError(
                "smooth_terms is an empty list, so no covariate would be modeled with "
                "splines. Pass at least one term, set smooth_terms=None to smooth all "
                "continuous covariates, or use method='fortin'/'chen'."
            )
        resolved: list[Any] = []
        for t in self.smooth_terms:
            if t in cont.columns:
                resolved.append(t)
            elif isinstance(t, int | np.integer) and 0 <= int(t) < len(cols):
                resolved.append(cols[int(t)])
            else:
                raise ValueError(
                    f"smooth_term {t!r} not found in continuous_covariates "
                    f"(columns: {cols}). Use a column name or a valid integer position."
                )
        return resolved

    def _term_bounds(self, col: Any, x: FloatArray) -> tuple[float, float]:
        """Boundary knots for a smooth term, from ``smooth_term_bounds`` or data."""
        b = self.smooth_term_bounds
        if b is None:
            return float(np.min(x)), float(np.max(x))
        if isinstance(b, dict):
            if col in b:
                lo, hi = b[col]
                return float(lo), float(hi)
            return float(np.min(x)), float(np.max(x))
        lo, hi = b
        return float(lo), float(hi)

    def _spline_knots(self, x: FloatArray, lo: float, hi: float) -> FloatArray:
        """Knot vector: ``degree+1`` boundary knots plus interior sample quantiles.

        Matches ``statsmodels.gam.smooth_basis.BSplines`` (and hence neuroHarmonize):
        ``spline_df - spline_degree - 1`` interior knots at equally spaced quantiles.
        """
        degree = self.spline_degree
        n_interior = self.spline_df - degree - 1
        if n_interior > 0:
            probs = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            interior = np.quantile(x, probs)
        else:
            interior = np.empty(0, dtype=np.float64)
        knots = np.concatenate([np.full(degree + 1, lo), interior, np.full(degree + 1, hi)])
        return np.asarray(knots, dtype=np.float64)

    def _spline_basis(self, x: FloatArray, knots: FloatArray) -> FloatArray:
        """B-spline design matrix with the first (identifiability) column dropped.

        Out-of-range values are clamped to the boundary knots (constant
        extrapolation), so held-out data never falls outside the basis support.
        """
        xc = np.clip(np.asarray(x, dtype=np.float64), knots[0], knots[-1])
        basis = BSpline.design_matrix(xc, knots, self.spline_degree).toarray()
        return np.asarray(basis[:, 1:], dtype=np.float64)

    def _apply_smooth_terms(self, cont: pd.DataFrame | None, *, fit: bool) -> pd.DataFrame:
        """Replace each smooth continuous column with its B-spline basis columns.

        Non-smooth columns stay linear. Knots are learned on ``fit`` and reused after.
        """
        if cont is None or cont.shape[1] == 0:
            raise ValueError(
                f"method={self.method!r} (ComBat-GAM) requires at least one continuous "
                f"covariate to model with splines, but `continuous_covariates` is empty. "
                f"Pass continuous covariates, or use method='fortin'/'chen' for a linear fit."
            )
        if fit:
            self._smooth_cols = self._resolve_smooth_columns(cont)
            self._smooth_knots = {}
            self._smooth_basis_columns = {}
        smooth_cols = set(self._smooth_cols)

        blocks: list[pd.DataFrame] = []
        for col in cont.columns:
            if col not in smooth_cols:
                blocks.append(cont[[col]].astype(float))
                continue
            x = cont[col].to_numpy(dtype=np.float64)
            if fit:
                if np.unique(x).size < self.spline_df:
                    raise ValueError(
                        f"Smooth term {col!r} has only {int(np.unique(x).size)} distinct "
                        f"value(s), too few for spline_df={self.spline_df} (need >= "
                        f"{self.spline_df}). Lower spline_df, drop it from smooth_terms, "
                        f"or model it linearly with method='fortin'/'chen'."
                    )
                lo, hi = self._term_bounds(col, x)
                if lo >= hi:
                    raise ValueError(
                        f"smooth_term_bounds for {col!r} must satisfy lo < hi, got ({lo}, {hi})."
                    )
                if x.min() < lo or x.max() > hi:
                    raise ValueError(
                        f"smooth_term_bounds ({lo}, {hi}) for {col!r} do not contain the "
                        f"training range [{x.min():.4g}, {x.max():.4g}]. Bounds must cover the "
                        f"full data range (set them wider to include held-out data); held-out "
                        f"values beyond the bounds are clamped at transform."
                    )
                knots = self._spline_knots(x, lo, hi)
                basis = self._spline_basis(x, knots)
                names = [f"{col}__spline{j}" for j in range(basis.shape[1])]
                self._smooth_knots[col] = knots
                self._smooth_basis_columns[col] = names
            else:
                knots = self._smooth_knots[col]
                basis = self._spline_basis(x, knots)
                names = self._smooth_basis_columns[col]
            blocks.append(pd.DataFrame(basis, index=cont.index, columns=names))
        return pd.concat(blocks, axis=1)

    def _fit_fortin(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> None:
        """Fortin et al. (2018) neuroComBat (and Pomponio et al. 2020 ComBat-GAM).

        For the GAM methods the smooth covariates are first replaced by B-spline bases.
        """
        if self._is_gam():
            cont = self._apply_smooth_terms(cont, fit=True)

        self._batch_levels = batch.cat.categories
        n_batch = len(self._batch_levels)
        n_samples = len(X)

        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)
        if self.reference_batch is not None:
            if self.reference_batch not in self._batch_levels:
                raise ValueError(
                    f"reference_batch={self.reference_batch!r} not found. "
                    f"Available batches: {list(self._batch_levels)}. "
                )
            batch_dummies.loc[:, self.reference_batch] = 1.0

        parts: list[pd.DataFrame] = [batch_dummies]
        if disc is not None:
            parts.append(self._discrete_dummies(disc, fit=True))

        if cont is not None:
            parts.append(cont.astype(float))

        design_df = pd.concat(parts, axis=1)
        self._nonbatch_columns = design_df.columns[n_batch:].tolist()
        design = design_df.values
        self._design_cond = float(la.cond(design))
        p_design = design.shape[1]
        self._design_rank = int(la.matrix_rank(design))
        self._design_ncols = p_design

        X_np = X.values
        beta_hat = la.lstsq(design, X_np, rcond=None)[0]

        beta_hat_batch = beta_hat[:n_batch]
        self._beta_hat_nonbatch = beta_hat[n_batch:]

        n_per_batch: FloatArray = np.asarray(
            batch.value_counts().sort_index().astype(int).values, dtype=np.float64
        )
        self._n_per_batch = dict(zip(self._batch_levels, n_per_batch.astype(int), strict=True))

        if self.reference_batch is not None:
            ref_idx = list(self._batch_levels).index(self.reference_batch)
            grand_mean = beta_hat_batch[ref_idx]
        else:
            grand_mean = (n_per_batch / n_samples) @ beta_hat_batch
            ref_idx = None

        self._grand_mean = pd.Series(grand_mean, index=X.columns)

        if self.reference_batch is not None:
            ref_mask = np.asarray(batch == self.reference_batch)
            resid = X_np[ref_mask] - design[ref_mask] @ beta_hat
            denom = int(ref_mask.sum())
        else:
            resid = X_np - design @ beta_hat
            denom = n_samples
        var_pooled = (resid**2).sum(axis=0) / denom + self.eps
        self._pooled_var = pd.Series(var_pooled, index=X.columns)

        stand_mean = grand_mean + design[:, n_batch:] @ self._beta_hat_nonbatch
        Xs = (X_np - stand_mean) / np.sqrt(var_pooled)

        gamma_hat = np.vstack([Xs[batch == lvl].mean(axis=0) for lvl in self._batch_levels])
        delta_hat = np.vstack(
            [Xs[batch == lvl].var(axis=0, ddof=1) + self.eps for lvl in self._batch_levels]
        )

        b_names = [str(lvl) for lvl in self._batch_levels]
        gamma_star, delta_star = self._shrink_eb(gamma_hat, delta_hat, n_per_batch, b_names)

        if ref_idx is not None:
            gamma_star[ref_idx] = 0.0
            if not self.mean_only:
                delta_star[ref_idx] = 1.0
        self._reference_batch_idx = ref_idx

        self._gamma_star = gamma_star
        self._delta_star = delta_star
        self._n_batch = n_batch
        self._p_design = p_design

    def _fit_chen(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> None:
        """Chen et al. (2022) CovBat, Steps 1-3 (also serves the covbat_gam method).

        The PCA runs on the Fortin residuals with the covariate effects removed, so the
        score correction cannot absorb an unbalanced covariate. Scores are moment-matched
        in mean and variance without EB (``score.eb = FALSE``), whatever ``mean_only``.
        """
        self._fit_fortin(X, batch, disc, cont)
        resid, _ = self._fortin_residuals(X, batch, disc, cont)
        pca = PCA(svd_solver="full", whiten=False).fit(resid)

        # PCs with ~0 training variance are dropped: standardizing them divides by ~0.
        ev = pca.explained_variance_
        n_with_variance = int(np.sum(ev > _PC_RELATIVE_VARIANCE_FLOOR * ev[0])) if ev[0] > 0 else 0

        if isinstance(self.covbat_cov_thresh, int):
            n_pc = min(self.covbat_cov_thresh, n_with_variance)
        else:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            # The cumulative ratio may end below 1.0 in floating point; cap it.
            n_pc = min(
                int(np.searchsorted(cumulative, self.covbat_cov_thresh) + 1), n_with_variance
            )

        self._covbat_pca = pca
        self._covbat_n_pc = n_pc

        if n_pc < 2:
            warnings.warn(
                f"CovBat retained fewer than two principal components (n_pc={n_pc}); covariance "
                f"harmonization is not meaningful with fewer than two, and the correction "
                f"reduces to the mean/variance adjustment already done by Fortin. Consider "
                f"method='fortin', or raise covbat_cov_thresh to retain more components.",
                stacklevel=2,
            )

        scores = pca.transform(resid)[:, :n_pc]
        masks = [np.asarray(batch == lvl) for lvl in self._batch_levels]
        within = scores.copy()
        for mask in masks:
            within[mask] -= scores[mask].mean(axis=0)
        self._pc_score_mean = scores.mean(axis=0)
        self._pc_score_var = (within**2).sum(axis=0) / len(scores) + self.eps
        z = (scores - self._pc_score_mean) / np.sqrt(self._pc_score_var)
        gamma_hat = np.vstack([z[mask].mean(axis=0) for mask in masks])
        delta_hat = np.vstack([z[mask].var(axis=0, ddof=1) + self.eps for mask in masks])

        self._pc_gamma, self._pc_delta, _ = self._apply_reference_shift(
            gamma_hat, delta_hat, self._batch_levels, mean_only=False
        )

    def _discrete_dummies(self, disc: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        """Dummy-code discrete covariates against the levels recorded at fit.

        Frozen levels keep ``transform`` a function of each sample alone, so a subset
        missing a level is not coded as the reference level.
        """
        if fit:
            self._discrete_levels = {col: _fit_levels(disc[col]) for col in disc.columns}
        coded = pd.DataFrame(index=disc.index)
        for col in disc.columns:
            if col not in self._discrete_levels:
                raise ValueError(
                    f"discrete covariate {col!r} was not present at fit. Pass the same "
                    f"covariate columns to transform as to fit. Seen at fit: "
                    f"{list(self._discrete_levels)}."
                )
            coded[col] = _code_levels(
                disc[col], self._discrete_levels[col], f"discrete_covariates column {col!r}"
            )
        return pd.get_dummies(coded, drop_first=True).astype(float)

    def _build_longitudinal_design(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
        time: pd.DataFrame | None,
        *,
        fit: bool,
    ) -> tuple[FloatArray, list[str]]:
        """Assemble the fixed-effects design [batch dummies | covariates | time].

        Returns the design matrix and the list of non-batch column names.
        """
        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)
        batch_dummies = batch_dummies.reindex(columns=self._batch_levels, fill_value=0.0)

        nonbatch_parts: list[pd.DataFrame] = []
        if disc is not None:
            nonbatch_parts.append(self._discrete_dummies(disc, fit=fit))
        if cont is not None:
            nonbatch_parts.append(cont.astype(float))
        if time is not None:
            nonbatch_parts.append(time.astype(float))

        if nonbatch_parts:
            nonbatch_df = pd.concat(nonbatch_parts, axis=1)
        else:
            nonbatch_df = pd.DataFrame(index=batch_dummies.index)

        design_df = pd.concat([batch_dummies, nonbatch_df], axis=1)
        n_batch = len(self._batch_levels)
        nonbatch_columns = design_df.columns[n_batch:].tolist()
        return design_df.values.astype(np.float64), nonbatch_columns

    def _fit_longitudinal(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
        subject_id: ArrayLike | None,
        time_covariate: ArrayLike | None,
    ) -> None:
        """Longitudinal ComBat (*Beer et al.* 2020).

        As Fortin, but the mean model has a per-subject random intercept (REML) whose
        BLUP is included in the standardization mean.
        """
        if subject_id is None:
            raise ValueError(
                "method='longitudinal' requires `subject_id` (one label per sample "
                "identifying the subject/individual used for the random intercept)."
            )

        self._batch_levels = batch.cat.categories
        n_batch = len(self._batch_levels)
        n_samples = len(X)

        counts = batch.value_counts()
        small = counts[counts < 2]
        if len(small):
            lvl = small.index[0]
            raise ValueError(
                f"Batch '{lvl}' has only {int(small.iloc[0])} sample(s). ComBat requires "
                f">= 2 samples per batch. Consider merging small batches or removing them."
            )

        time = self._to_df(time_covariate, X.index, "time_covariate")
        design, self._nonbatch_columns = self._build_longitudinal_design(
            X, batch, disc, cont, time, fit=True
        )
        self._design_cond = float(la.cond(design)) if design.shape[1] else 1.0
        self._design_rank = int(la.matrix_rank(design)) if design.shape[1] else None
        self._design_ncols = design.shape[1] if design.shape[1] else None
        self._n_batch = n_batch
        self._p_design = design.shape[1]

        subj = self._as_series(subject_id, X.index, "subject_id")
        self._re_subject_levels = subj.cat.categories
        group_idx = subj.cat.codes.to_numpy().astype(np.intp)
        n_groups = len(self._re_subject_levels)
        n_k = np.bincount(group_idx, minlength=n_groups).astype(np.float64)

        beta, blup, sigma2, _ = fit_random_intercept(
            design, X.values.astype(np.float64), group_idx, n_groups, n_k, eps=self.eps
        )
        self._re_blup = blup

        beta_batch = beta[:n_batch]
        self._beta_hat_nonbatch = beta[n_batch:]

        n_per_batch_arr = np.asarray(
            batch.value_counts().reindex(self._batch_levels).astype(int).values, dtype=np.float64
        )
        self._n_per_batch = dict(zip(self._batch_levels, n_per_batch_arr.astype(int), strict=True))

        grand_mean = (n_per_batch_arr / n_samples) @ beta_batch
        self._grand_mean = pd.Series(grand_mean, index=X.columns)
        var_pooled = sigma2 + self.eps
        self._pooled_var = pd.Series(var_pooled, index=X.columns)

        blup_per_row = blup[group_idx]
        stand_mean = grand_mean + design[:, n_batch:] @ self._beta_hat_nonbatch + blup_per_row
        Xs = (X.values - stand_mean) / np.sqrt(var_pooled)

        gamma_hat = np.vstack(
            [Xs[(batch == lvl).to_numpy()].mean(axis=0) for lvl in self._batch_levels]
        )
        delta_hat = np.vstack(
            [
                Xs[(batch == lvl).to_numpy()].var(axis=0, ddof=1) + self.eps
                for lvl in self._batch_levels
            ]
        )

        b_names = [str(lvl) for lvl in self._batch_levels]
        gamma_star, delta_star = self._shrink_eb(gamma_hat, delta_hat, n_per_batch_arr, b_names)
        gamma_star, delta_star, self._reference_batch_idx = self._apply_reference_shift(
            gamma_star,
            delta_star,
            self._batch_levels,
            gamma_hat=gamma_hat,
            delta_hat=delta_hat,
        )

        self._gamma_star = gamma_star
        self._delta_star = delta_star

    def _shrink_gamma_delta(
        self,
        gamma_hat: FloatArray,
        delta_hat: FloatArray,
        n_per_batch: dict[str, int] | FloatArray,
        *,
        parametric: bool,
        max_iter: int = 100,
        tol: float = 1e-4,
        batch_names: list[str] | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Empirical Bayes shrinkage estimation (Johnson et al. 2007).

        Hyperpriors are per batch, pooled over units, as in ``sva::ComBat``; the
        non-parametric branch is ``sva``'s ``int.eprior``. With fewer than two units
        the prior is non-identifiable and the per-unit MLE is returned.
        """
        B, p = gamma_hat.shape
        n_vec = (
            np.array(list(n_per_batch.values())) if isinstance(n_per_batch, dict) else n_per_batch
        )
        names = [batch_names[i] if batch_names else str(i) for i in range(B)]

        def closed_form_info() -> list[dict[str, object]]:
            return [
                {
                    "batch": name,
                    "converged": True,
                    "iterations": 0,
                    "final_gamma_change": 0.0,
                    "final_delta_change": 0.0,
                    "closed_form": True,
                }
                for name in names
            ]

        if p < 2:
            self._convergence_info = closed_form_info()
            return gamma_hat.copy(), delta_hat.copy()

        if parametric:
            gamma_bar = gamma_hat.mean(axis=1)
            t2 = np.maximum(gamma_hat.var(axis=1, ddof=1), self.eps)

            def postmean(
                g_hat: FloatArray,
                g_bar: FloatArray,
                n: float,
                d_star: FloatArray,
                t2_: FloatArray,
            ) -> FloatArray:
                return (t2_ * n * g_hat + d_star * g_bar) / (t2_ * n + d_star)

            def postvar(
                sum2: FloatArray, n: float, a: FloatArray | float, b: FloatArray | float
            ) -> FloatArray:
                return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

            convergence_info: list[dict[str, object]] = []
            delta_mean = delta_hat.mean(axis=1)
            delta_var = np.maximum(delta_hat.var(axis=1, ddof=1), self.eps)
            a_prior = delta_mean**2 / delta_var + 2
            b_prior = delta_mean * (a_prior - 1)

            gamma_star = np.empty_like(gamma_hat)
            delta_star = np.empty_like(delta_hat)

            for i in range(B):
                n_i = n_vec[i]
                g_hat_i = gamma_hat[i]
                d_hat_i = np.maximum(delta_hat[i], self.eps)

                g_new = postmean(g_hat_i, gamma_bar[i], n_i, d_hat_i, t2[i])
                sum2 = (n_i - 1) * d_hat_i + n_i * (g_hat_i - g_new) ** 2
                d_new = postvar(sum2, n_i, a_prior[i], b_prior[i])

                converged = False
                n_iter = 0
                gamma_change = float("inf")
                delta_change = float("inf")
                for n_iter in range(1, max_iter + 1):  # noqa: B007
                    g_prev, d_prev = g_new, d_new
                    g_new = postmean(g_hat_i, gamma_bar[i], n_i, d_prev, t2[i])
                    sum2 = (n_i - 1) * d_hat_i + n_i * (g_hat_i - g_new) ** 2
                    d_new = postvar(sum2, n_i, a_prior[i], b_prior[i])
                    gamma_change = float(
                        np.max(np.abs(g_new - g_prev) / (np.abs(g_prev) + self.eps))
                    )
                    delta_change = float(
                        np.max(np.abs(d_new - d_prev) / (np.abs(d_prev) + self.eps))
                    )
                    if gamma_change < tol and (self.mean_only or delta_change < tol):
                        converged = True
                        break

                b_name = names[i]
                if not converged:
                    warnings.warn(
                        f"Parametric EB did not converge for batch '{b_name}' "
                        f"within {max_iter} iterations "
                        f"(final relative change: {max(gamma_change, delta_change):.2e}, "
                        f"tol: {tol:.1e}). Results may be unreliable for this batch.",
                        stacklevel=2,
                    )
                convergence_info.append(
                    {
                        "batch": b_name,
                        "converged": converged,
                        "iterations": n_iter,
                        "final_gamma_change": gamma_change,
                        "final_delta_change": delta_change,
                        "closed_form": False,
                    }
                )
                gamma_star[i] = g_new
                delta_star[i] = d_new
            self._convergence_info = convergence_info
            return gamma_star, delta_star

        # Log-space weights: a large batch would otherwise underflow them all.
        gamma_star = np.empty_like(gamma_hat)
        delta_star = np.empty_like(delta_hat)
        block = max(1, _EPRIOR_BLOCK_ELEMENTS // p)
        for i in range(B):
            n_i = float(n_vec[i])
            g = gamma_hat[i]
            d = np.maximum(delta_hat[i], self.eps)
            log_norm = -0.5 * n_i * np.log(2.0 * np.pi * d)
            for start in range(0, p, block):
                stop = min(start + block, p)
                own = np.arange(start, stop)
                sum2 = (n_i - 1.0) * d[own, None] + n_i * (g[own, None] - g[None, :]) ** 2
                log_lh = log_norm[None, :] - sum2 / (2.0 * d[None, :])
                log_lh[own - start, own] = -np.inf
                weights = np.exp(log_lh - log_lh.max(axis=1, keepdims=True))
                weights /= weights.sum(axis=1, keepdims=True)
                gamma_star[i, start:stop] = weights @ g
                delta_star[i, start:stop] = weights @ d
        self._convergence_info = closed_form_info()
        return gamma_star, delta_star

    def _shrink_gamma(
        self,
        gamma_hat: FloatArray,
        delta_hat: FloatArray,
        n_per_batch: dict[str, int] | FloatArray,
        *,
        parametric: bool,
        batch_names: list[str] | None = None,
    ) -> FloatArray:
        """Convenience wrapper that returns only gamma* (for *mean-only* mode)."""
        gamma, _ = self._shrink_gamma_delta(
            gamma_hat,
            delta_hat,
            n_per_batch,
            parametric=parametric,
            batch_names=batch_names,
        )
        return gamma

    def _shrink_eb(
        self,
        gamma_hat: FloatArray,
        delta_hat: FloatArray,
        n_per_batch: dict[str, int] | FloatArray,
        batch_names: list[str],
    ) -> tuple[FloatArray, FloatArray]:
        """Empirical-Bayes shrinkage honoring ``mean_only`` (returns gamma*, delta*)."""
        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat,
                delta_hat,
                n_per_batch,
                parametric=self.parametric,
                batch_names=batch_names,
            )
            delta_star = np.ones_like(delta_hat)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat,
                delta_hat,
                n_per_batch,
                parametric=self.parametric,
                batch_names=batch_names,
            )
        return gamma_star, delta_star

    def _apply_reference_shift(
        self,
        gamma_star: FloatArray,
        delta_star: FloatArray,
        batch_levels: pd.Index,
        *,
        gamma_hat: FloatArray | None = None,
        delta_hat: FloatArray | None = None,
        mean_only: bool | None = None,
    ) -> tuple[FloatArray, FloatArray, int | None]:
        """Re-express gamma*/delta* relative to the reference batch.

        The target is the reference batch's own unshrunk ``gamma_hat``/``delta_hat`` when
        given, as in ``sva::ComBat``. Returns the shifted arrays and the reference index
        (``None`` without one). ``mean_only`` defaults to the estimator's. Fortin uses
        its design instead.
        """
        if mean_only is None:
            mean_only = self.mean_only
        if self.reference_batch is None:
            return gamma_star, delta_star, None
        ref_idx = list(batch_levels).index(self.reference_batch)
        gamma_ref = (gamma_star if gamma_hat is None else gamma_hat)[ref_idx]
        delta_ref = (delta_star if delta_hat is None else delta_hat)[ref_idx]
        if mean_only:
            gamma_star = gamma_star - gamma_ref
        else:
            # Transforms apply (x - g') / sqrt(d'), so gamma_ref scales by sqrt(d').
            ratio = delta_star / delta_ref
            gamma_star = gamma_star - gamma_ref * np.sqrt(ratio)
            delta_star = ratio
        return gamma_star, delta_star, ref_idx

    def transform(
        self,
        X: ArrayLike,
        *,
        batch: ArrayLike,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        subject_id: ArrayLike | None = None,
        time_covariate: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Transform the data using fitted ComBat parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        batch : array-like of shape (n_samples,)
            Batch labels for each sample.
        discrete_covariates : array-like or None, default=None
            Categorical covariates (Fortin/Chen/Longitudinal only).
        continuous_covariates : array-like or None, default=None
            Continuous covariates (Fortin/Chen/Longitudinal only).
        subject_id : array-like or None, default=None
            Subject labels for the random intercept (Longitudinal only). Unseen
            subjects fall back to a zero random intercept.
        time_covariate : array-like or None, default=None
            Continuous time variable (Longitudinal only).

        Returns
        -------
        X_adjusted : pd.DataFrame
            Batch-corrected data.

        Raises
        ------
        ValueError
            If the model is not fitted or if unseen batch levels are present.
        """
        if not hasattr(self, "_gamma_star"):
            raise ValueError(
                "This ComBatModel instance is not fitted yet. Call 'fit' before 'transform'."
            )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        idx = X.index
        batch = self._as_series(batch, idx, "batch")
        unseen = set(batch.cat.categories) - set(self._batch_levels)
        if unseen:
            raise ValueError(f"Unseen batch levels during transform: {unseen}.")
        disc = self._to_df(discrete_covariates, idx, "discrete_covariates")
        cont = self._to_df(continuous_covariates, idx, "continuous_covariates")

        method = _resolve_method(self.method)
        if method == "johnson":
            return self._transform_johnson(X, batch)
        elif method in {"fortin", "gam"}:
            return self._transform_fortin(X, batch, disc, cont)
        elif method in {"chen", "covbat_gam"}:
            return self._transform_chen(X, batch, disc, cont)
        elif method == "longitudinal":
            return self._transform_longitudinal(X, batch, disc, cont, subject_id, time_covariate)
        else:
            raise ValueError(
                f"Unknown method={self.method!r}. Expected an author name "
                f"('johnson', 'fortin', 'chen', 'longitudinal', 'gam', 'covbat_gam') or a "
                f"method-name alias ('classic_combat', 'neurocombat', 'covbat', "
                f"'longcombat', 'combat_gam')."
            )

    def _transform_johnson(self, X: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
        """Johnson transform implementation."""
        pooled = self._pooled_var
        grand = self._grand_mean

        Xs = (X - grand) / np.sqrt(pooled)
        X_adj = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)

        for i, lvl in enumerate(self._batch_levels):
            idx = batch == lvl
            if not idx.any():
                continue
            if self.reference_batch is not None and lvl == self.reference_batch:
                X_adj.loc[idx] = X.loc[idx].values
                continue

            g = self._gamma_star[i]
            d = self._delta_star[i]
            Xb = Xs.loc[idx] - g if self.mean_only else (Xs.loc[idx] - g) / np.sqrt(d)
            X_adj.loc[idx] = (Xb * np.sqrt(pooled) + grand).values
        return X_adj

    def _transform_fortin(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Fortin transform implementation (also serves the gam method)."""
        resid, stand_mu = self._fortin_residuals(X, batch, disc, cont)
        return pd.DataFrame(resid + stand_mu, index=X.index, columns=X.columns, dtype=float)

    def _fortin_residuals(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> tuple[FloatArray, FloatArray]:
        """Return ``(resid, stand_mu)``, whose sum is the Fortin-harmonized data.

        ``stand_mu`` is the intercept plus the protected covariate effects, so
        ``resid`` is the ComBat-adjusted residual of Chen et al. (2022), Step 1.
        """
        if self._is_gam():
            cont = self._apply_smooth_terms(cont, fit=False)

        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)
        batch_dummies = batch_dummies.reindex(columns=self._batch_levels, fill_value=0.0)
        if self.reference_batch is not None:
            batch_dummies.loc[:, self.reference_batch] = 1.0

        nonbatch_parts: list[pd.DataFrame] = []
        if disc is not None:
            nonbatch_parts.append(self._discrete_dummies(disc, fit=False))
        if cont is not None:
            nonbatch_parts.append(cont.astype(float))

        if nonbatch_parts:
            nonbatch_df = pd.concat(nonbatch_parts, axis=1)
        else:
            nonbatch_df = pd.DataFrame(index=batch_dummies.index)
        nonbatch_df = nonbatch_df.reindex(columns=self._nonbatch_columns, fill_value=0.0)

        design = pd.concat([batch_dummies, nonbatch_df], axis=1).values

        X_np = X.values
        grand_vals = np.asarray(self._grand_mean.values, dtype=np.float64)
        pooled_vals = np.asarray(self._pooled_var.values, dtype=np.float64)
        stand_mu = grand_vals + design[:, self._n_batch :] @ self._beta_hat_nonbatch
        Xs = (X_np - stand_mu) / np.sqrt(pooled_vals)

        for i, lvl in enumerate(self._batch_levels):
            idx = batch == lvl
            if not idx.any():
                continue
            if self.reference_batch is not None and lvl == self.reference_batch:
                continue

            g = self._gamma_star[i]
            d = self._delta_star[i]
            if self.mean_only:
                Xs[idx] = Xs[idx] - g
            else:
                Xs[idx] = (Xs[idx] - g) / np.sqrt(d)

        resid: FloatArray = Xs * np.sqrt(pooled_vals)
        return resid, stand_mu

    def _transform_chen(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Chen et al. (2022) Step 4 (also serves the covbat_gam method).

        Only the change in the retained scores is added to the residual, so a held-out
        row keeps the part outside the training PC span and the reference is untouched.
        """
        resid, stand_mu = self._fortin_residuals(X, batch, disc, cont)
        n_pc = self._covbat_n_pc
        retained = self._covbat_pca.components_[:n_pc]
        scores = self._covbat_pca.transform(resid)[:, :n_pc]
        score_sd = np.sqrt(self._pc_score_var)
        z = (scores - self._pc_score_mean) / score_sd
        z_adj = z.copy()

        for i, lvl in enumerate(self._batch_levels):
            idx = np.asarray(batch == lvl)
            if not idx.any():
                continue
            if self.reference_batch is not None and lvl == self.reference_batch:
                continue
            z_adj[idx] = (z[idx] - self._pc_gamma[i]) / np.sqrt(self._pc_delta[i])

        resid_covbat = resid + ((z_adj - z) * score_sd) @ retained
        return pd.DataFrame(resid_covbat + stand_mu, index=X.index, columns=X.columns)

    def _transform_longitudinal(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
        subject_id: ArrayLike | None,
        time_covariate: ArrayLike | None,
    ) -> pd.DataFrame:
        """Longitudinal transform implementation."""
        if subject_id is None:
            raise ValueError("method='longitudinal' requires `subject_id` at transform time.")

        n_features = len(self._grand_mean)
        time = self._to_df(time_covariate, X.index, "time_covariate")
        design, nonbatch_columns = self._build_longitudinal_design(
            X, batch, disc, cont, time, fit=False
        )
        nonbatch_design = pd.DataFrame(
            design[:, self._n_batch :],
            index=X.index,
            columns=nonbatch_columns,
        ).reindex(columns=self._nonbatch_columns, fill_value=0.0)

        subj = self._as_series(subject_id, X.index, "subject_id")
        level_pos = {lvl: i for i, lvl in enumerate(self._re_subject_levels)}
        codes = np.array([level_pos.get(s, -1) for s in subj.to_numpy()], dtype=np.intp)
        blup_per_row = np.zeros((len(X.index), n_features), dtype=np.float64)
        seen = codes >= 0
        if seen.any():
            blup_per_row[seen] = self._re_blup[codes[seen]]

        grand_vals = np.asarray(self._grand_mean.values, dtype=np.float64)
        pooled_vals = np.asarray(self._pooled_var.values, dtype=np.float64)
        stand_mean = grand_vals + nonbatch_design.values @ self._beta_hat_nonbatch + blup_per_row
        Xs = (X.values - stand_mean) / np.sqrt(pooled_vals)

        for i, lvl in enumerate(self._batch_levels):
            idx = (batch == lvl).to_numpy()
            if not idx.any():
                continue
            if self.reference_batch is not None and lvl == self.reference_batch:
                continue
            g = self._gamma_star[i]
            d = self._delta_star[i]
            if self.mean_only:
                Xs[idx] = Xs[idx] - g
            else:
                Xs[idx] = (Xs[idx] - g) / np.sqrt(d)

        X_adj = Xs * np.sqrt(pooled_vals) + stand_mean
        return pd.DataFrame(X_adj, index=X.index, columns=X.columns, dtype=float)

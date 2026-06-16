"""ComBat algorithm core.

`ComBatModel` implements four variants of the ComBat algorithm:
    * Johnson et al. (2007) vanilla ComBat (method="johnson")
    * Fortin et al. (2018) extension with covariates (method="fortin")
    * Chen et al. (2022) CovBat (method="chen")
    * Beer et al. (2020) Longitudinal ComBat (method="longitudinal")
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA

from ._mixed import fit_random_intercept

ArrayLike: TypeAlias = pd.DataFrame | pd.Series | npt.NDArray[Any]
FloatArray: TypeAlias = npt.NDArray[np.float64]


class ComBatModel:
    """ComBat algorithm.

    Parameters
    ----------
    method : {'johnson', 'fortin', 'chen', 'longitudinal'}, default='johnson'
        * 'johnson' - classic ComBat.
        * 'fortin' - covariate-aware ComBat.
        * 'chen' - CovBat, PCA-based ComBat.
        * 'longitudinal' - Longitudinal ComBat; requires a per-sample
          ``subject_id`` for the random intercept.
    parametric : bool, default=True
        Use the parametric empirical Bayes variant.
    mean_only : bool, default=False
        If True, only the mean is adjusted (`gamma_star`),
        ignoring the variance (`delta_star`).
    reference_batch : str, optional
        If specified, the batch level to use as reference.
    covbat_cov_thresh : float or int, default=0.9
        CovBat: cumulative variance threshold (0, 1] to retain PCs, or
        integer >= 1 specifying the number of components directly.
    eps : float, default=1e-8
        Numerical jitter to avoid division-by-zero.
    """

    def __init__(
        self,
        *,
        method: Literal["johnson", "fortin", "chen", "longitudinal"] = "johnson",
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: str | None = None,
        eps: float = 1e-8,
        covbat_cov_thresh: float | int = 0.9,
    ) -> None:
        self.method: str = method
        self.parametric: bool = parametric
        self.mean_only: bool = bool(mean_only)
        self.reference_batch: str | None = reference_batch
        self.eps: float = float(eps)
        self.covbat_cov_thresh: float | int = covbat_cov_thresh

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
        self._batch_levels_pc: pd.Index
        self._pc_gamma_star: FloatArray
        self._pc_delta_star: FloatArray
        self._re_subject_levels: pd.Index
        self._re_blup: FloatArray

        # Validate covbat_cov_thresh
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
        method = self.method.lower()
        if method not in {"johnson", "fortin", "chen", "longitudinal"}:
            raise ValueError(
                f"method={self.method!r} is not recognized. "
                f"Expected one of 'johnson', 'fortin', 'chen', or 'longitudinal'."
            )
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
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
        elif method == "fortin":
            self._fit_fortin(X, batch, disc, cont)
        elif method == "chen":
            self._fit_chen(X, batch, disc, cont)
        elif method == "longitudinal":
            self._fit_longitudinal(X, batch, disc, cont, subject_id, time_covariate)
        return self

    def _fit_johnson(self, X: pd.DataFrame, batch: pd.Series) -> None:
        """Johnson et al. (2007) ComBat."""
        self._batch_levels = batch.cat.categories
        grand_mean = X.mean(axis=0)

        # Within-batch residual variance
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
        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat_arr,
                delta_hat_arr,
                n_per_batch,
                parametric=self.parametric,
                batch_names=b_names,
            )
            delta_star = np.ones_like(delta_hat_arr)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat_arr,
                delta_hat_arr,
                n_per_batch,
                parametric=self.parametric,
                batch_names=b_names,
            )

        if self.reference_batch is not None:
            ref_idx = list(self._batch_levels).index(self.reference_batch)
            gamma_ref = gamma_star[ref_idx]
            delta_ref = delta_star[ref_idx]
            gamma_star = gamma_star - gamma_ref
            if not self.mean_only:
                delta_star = delta_star / delta_ref
            self._reference_batch_idx = ref_idx
        else:
            self._reference_batch_idx = None

        self._grand_mean = grand_mean
        self._pooled_var = pooled_var
        self._gamma_star = gamma_star
        self._delta_star = delta_star
        self._n_per_batch = n_per_batch
        self._design_cond: float | None = None

    def _fit_fortin(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> None:
        """Fortin et al. (2018) neuroComBat."""
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
            parts.append(pd.get_dummies(disc.astype("category"), drop_first=True).astype(float))

        if cont is not None:
            parts.append(cont.astype(float))

        design_df = pd.concat(parts, axis=1)
        self._nonbatch_columns = design_df.columns[n_batch:].tolist()
        design = design_df.values
        self._design_cond = float(la.cond(design))
        p_design = design.shape[1]

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
        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat,
                delta_hat,
                n_per_batch,
                parametric=self.parametric,
                batch_names=b_names,
            )
            delta_star = np.ones_like(delta_hat)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat,
                delta_hat,
                n_per_batch,
                parametric=self.parametric,
                batch_names=b_names,
            )

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
        """Chen et al. (2022) CovBat."""
        self._fit_fortin(X, batch, disc, cont)
        X_meanvar_adj = self._transform_fortin(X, batch, disc, cont)
        pca = PCA(svd_solver="full", whiten=False).fit(X_meanvar_adj)

        # Determine number of components based on threshold type
        if isinstance(self.covbat_cov_thresh, int):
            n_pc = min(self.covbat_cov_thresh, len(pca.explained_variance_ratio_))
        else:
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            n_pc = int(np.searchsorted(cumulative, self.covbat_cov_thresh) + 1)

        self._covbat_pca = pca
        self._covbat_n_pc = n_pc

        scores = pca.transform(X_meanvar_adj)[:, :n_pc]
        scores_df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i + 1}" for i in range(n_pc)])
        self._batch_levels_pc = self._batch_levels
        n_per_batch = self._n_per_batch

        gamma_hat: list[npt.NDArray[np.float64]] = []
        delta_hat: list[npt.NDArray[np.float64]] = []
        for lvl in self._batch_levels_pc:
            idx = batch == lvl
            xb = scores_df.loc[idx]
            gamma_hat.append(xb.mean(axis=0).values)
            delta_hat.append(xb.var(axis=0, ddof=1).values + self.eps)
        gamma_hat_arr = np.vstack(gamma_hat)
        delta_hat_arr = np.vstack(delta_hat)

        b_names = [str(lvl) for lvl in self._batch_levels_pc]
        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat_arr,
                delta_hat_arr,
                n_per_batch,
                parametric=self.parametric,
                batch_names=b_names,
            )
            delta_star = np.ones_like(delta_hat_arr)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat_arr,
                delta_hat_arr,
                n_per_batch,
                parametric=self.parametric,
                batch_names=b_names,
            )

        if self.reference_batch is not None:
            ref_idx = list(self._batch_levels_pc).index(self.reference_batch)
            gamma_ref = gamma_star[ref_idx]
            delta_ref = delta_star[ref_idx]
            gamma_star = gamma_star - gamma_ref
            if not self.mean_only:
                delta_star = delta_star / delta_ref

        self._pc_gamma_star = gamma_star
        self._pc_delta_star = delta_star

    def _build_longitudinal_design(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
        time: pd.DataFrame | None,
    ) -> tuple[FloatArray, list[str]]:
        """Assemble the fixed-effects design [batch dummies | covariates | time].

        Returns the design matrix and the list of non-batch column names.
        """
        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)
        batch_dummies = batch_dummies.reindex(columns=self._batch_levels, fill_value=0.0)

        nonbatch_parts: list[pd.DataFrame] = []
        if disc is not None:
            nonbatch_parts.append(
                pd.get_dummies(disc.astype("category"), drop_first=True).astype(float)
            )
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

        Identical to Fortin's standardize -> per-batch gamma/delta -> empirical
        Bayes -> remove pipeline, except the fixed-effects mean model is fit with
        a per-subject random intercept (REML), and the random-intercept BLUP is
        included in the standardization mean.
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
        design, self._nonbatch_columns = self._build_longitudinal_design(X, batch, disc, cont, time)
        self._design_cond = float(la.cond(design)) if design.shape[1] else 1.0
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
        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat,
                delta_hat,
                n_per_batch_arr,
                parametric=self.parametric,
                batch_names=b_names,
            )
            delta_star = np.ones_like(delta_hat)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat,
                delta_hat,
                n_per_batch_arr,
                parametric=self.parametric,
                batch_names=b_names,
            )

        if self.reference_batch is not None:
            ref_idx = list(self._batch_levels).index(self.reference_batch)
            gamma_ref = gamma_star[ref_idx]
            delta_ref = delta_star[ref_idx]
            gamma_star = gamma_star - gamma_ref
            if not self.mean_only:
                delta_star = delta_star / delta_ref
            self._reference_batch_idx = ref_idx
        else:
            self._reference_batch_idx = None

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
        """Empirical Bayes shrinkage estimation.

        Both the parametric and non-parametric branches use an iterative
        scheme that alternates between updating gamma (location) and delta
        (scale) posteriors until convergence.
        """
        B, _p = gamma_hat.shape
        n_vec = (
            np.array(list(n_per_batch.values())) if isinstance(n_per_batch, dict) else n_per_batch
        )
        gamma_bar = gamma_hat.mean(axis=0)
        t2 = np.maximum(gamma_hat.var(axis=0, ddof=1), self.eps)
        convergence_info: list[dict[str, object]] = []

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

        if parametric:
            delta_var = np.maximum(delta_hat.var(axis=0, ddof=1), self.eps)
            a_prior = (delta_hat.mean(axis=0) ** 2) / delta_var + 2
            b_prior = delta_hat.mean(axis=0) * (a_prior - 1)

            gamma_star = np.empty_like(gamma_hat)
            delta_star = np.empty_like(delta_hat)

            for i in range(B):
                n_i = n_vec[i]
                g_hat_i = gamma_hat[i]
                d_hat_i = np.maximum(delta_hat[i], self.eps)

                g_new = postmean(g_hat_i, gamma_bar, n_i, d_hat_i, t2)
                sum2 = (n_i - 1) * d_hat_i + n_i * (g_hat_i - g_new) ** 2
                d_new = postvar(sum2, n_i, a_prior, b_prior)

                converged = False
                n_iter = 0
                gamma_change = float("inf")
                delta_change = float("inf")
                for n_iter in range(1, max_iter + 1):  # noqa: B007  # used after loop
                    g_prev, d_prev = g_new, d_new
                    g_new = postmean(g_hat_i, gamma_bar, n_i, d_prev, t2)
                    sum2 = (n_i - 1) * d_hat_i + n_i * (g_hat_i - g_new) ** 2
                    d_new = postvar(sum2, n_i, a_prior, b_prior)
                    gamma_change = float(
                        np.max(np.abs(g_new - g_prev) / (np.abs(g_prev) + self.eps))
                    )
                    delta_change = float(
                        np.max(np.abs(d_new - d_prev) / (np.abs(d_prev) + self.eps))
                    )
                    if gamma_change < tol and (self.mean_only or delta_change < tol):
                        converged = True
                        break

                b_name = batch_names[i] if batch_names else str(i)
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
                    }
                )
                gamma_star[i] = g_new
                delta_star[i] = 1.0 if self.mean_only else d_new
            self._convergence_info = convergence_info
            return gamma_star, delta_star

        else:

            def aprior(delta: FloatArray) -> float:
                m = float(delta.mean())
                s2 = float(max(delta.var(ddof=1), self.eps))
                return (2 * s2 + m**2) / s2

            def bprior(delta: FloatArray) -> float:
                m = float(delta.mean())
                s2 = float(max(delta.var(ddof=1), self.eps))
                return (m * s2 + m**3) / s2

            gamma_star = np.empty_like(gamma_hat)
            delta_star = np.empty_like(delta_hat)

            for i in range(B):
                n_i = n_vec[i]
                g_hat_i = gamma_hat[i]
                d_hat_i = delta_hat[i]
                a_i = aprior(d_hat_i)
                b_i = bprior(d_hat_i)

                converged = False
                n_iter = 0
                gamma_change = float("inf")
                delta_change = float("inf")
                g_new, d_new = g_hat_i.copy(), d_hat_i.copy()
                for n_iter in range(1, max_iter + 1):  # noqa: B007  # used after loop
                    g_prev, d_prev = g_new, d_new
                    g_new = postmean(g_hat_i, gamma_bar, n_i, d_prev, t2)
                    sum2 = (n_i - 1) * d_hat_i + n_i * (g_hat_i - g_new) ** 2
                    d_new = postvar(sum2, n_i, a_i, b_i)
                    gamma_change = float(
                        np.max(np.abs(g_new - g_prev) / (np.abs(g_prev) + self.eps))
                    )
                    delta_change = float(
                        np.max(np.abs(d_new - d_prev) / (np.abs(d_prev) + self.eps))
                    )
                    if gamma_change < tol and (self.mean_only or delta_change < tol):
                        converged = True
                        break

                b_name = batch_names[i] if batch_names else str(i)
                if not converged:
                    warnings.warn(
                        f"Non-parametric EB did not converge for batch '{b_name}' "
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
                    }
                )
                gamma_star[i] = g_new
                delta_star[i] = 1.0 if self.mean_only else d_new
            self._convergence_info = convergence_info
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

        method = self.method.lower()
        if method == "johnson":
            return self._transform_johnson(X, batch)
        elif method == "fortin":
            return self._transform_fortin(X, batch, disc, cont)
        elif method == "chen":
            return self._transform_chen(X, batch, disc, cont)
        elif method == "longitudinal":
            return self._transform_longitudinal(X, batch, disc, cont, subject_id, time_covariate)
        else:
            raise ValueError(
                f"Unknown method={method!r}. Expected one of 'johnson', 'fortin', "
                f"'chen', or 'longitudinal'."
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
        """Fortin transform implementation."""
        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)
        batch_dummies = batch_dummies.reindex(columns=self._batch_levels, fill_value=0.0)
        if self.reference_batch is not None:
            batch_dummies.loc[:, self.reference_batch] = 1.0

        nonbatch_parts: list[pd.DataFrame] = []
        if disc is not None:
            nonbatch_parts.append(
                pd.get_dummies(disc.astype("category"), drop_first=True).astype(float)
            )
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
                # leave reference samples unchanged
                continue

            g = self._gamma_star[i]
            d = self._delta_star[i]
            if self.mean_only:
                Xs[idx] = Xs[idx] - g
            else:
                Xs[idx] = (Xs[idx] - g) / np.sqrt(d)

        X_adj = Xs * np.sqrt(pooled_vals) + stand_mu
        return pd.DataFrame(X_adj, index=X.index, columns=X.columns, dtype=float)

    def _transform_chen(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: pd.DataFrame | None,
        cont: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Chen transform implementation."""
        X_meanvar_adj = self._transform_fortin(X, batch, disc, cont)
        scores = self._covbat_pca.transform(X_meanvar_adj)
        n_pc = self._covbat_n_pc
        scores_adj = scores.copy()

        for i, lvl in enumerate(self._batch_levels_pc):
            idx = batch == lvl
            if not idx.any():
                continue
            if self.reference_batch is not None and lvl == self.reference_batch:
                continue
            g = self._pc_gamma_star[i]
            d = self._pc_delta_star[i]
            if self.mean_only:
                scores_adj[idx, :n_pc] = scores_adj[idx, :n_pc] - g
            else:
                scores_adj[idx, :n_pc] = (scores_adj[idx, :n_pc] - g) / np.sqrt(d)

        X_recon = self._covbat_pca.inverse_transform(scores_adj)
        return pd.DataFrame(X_recon, index=X.index, columns=X.columns)

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
        design, nonbatch_columns = self._build_longitudinal_design(X, batch, disc, cont, time)
        # Realign non-batch columns to those seen at fit (handles unseen covariate levels).
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

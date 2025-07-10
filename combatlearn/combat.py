"""ComBat algorithm.

`ComBatModel` implements both:
    * Johnson et al. (2007) vanilla ComBat (method="johnson")
    * Fortin et al. (2018) extension with covariates (method="fortin")
    * Chen et al. (2022) CovBat (method="chen")

`ComBat` makes the model compatible with scikit-learn by stashing
the batch (and optional covariates) at construction.
"""
from __future__ import annotations

import numpy as np
import numpy.linalg as la
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.decomposition import PCA
from typing import Literal, Optional, Union, Dict, Tuple, Any, cast
import numpy.typing as npt
import warnings

__author__ = "Ettore Rocchi"

ArrayLike = Union[pd.DataFrame, pd.Series, npt.NDArray[Any]]
FloatArray = npt.NDArray[np.float64]


class ComBatModel:
    """ComBat algorithm.

    Parameters
    ----------
    method : {'johnson', 'fortin', 'chen'}, default='johnson'
        * 'johnson' - classic ComBat.
        * 'fortin' - covariate-aware ComBat.
        * 'chen' - CovBat, PCA-based ComBat.
    parametric : bool, default=True
        Use the parametric empirical Bayes variant.
    mean_only : bool, default=False
        If True, only the mean is adjusted (`gamma_star`),
        ignoring the variance (`delta_star`).
    reference_batch : str, optional
        If specified, the batch level to use as reference.
    covbat_cov_thresh : float, default=0.9
        CovBat: cumulative explained variance threshold for PCA.
    eps : float, default=1e-8
        Numerical jitter to avoid division-by-zero.
    """

    def __init__(
        self,
        *,
        method: Literal["johnson", "fortin", "chen"] = "johnson", 
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: Optional[str] = None,
        eps: float = 1e-8,
        covbat_cov_thresh: float = 0.9,
    ) -> None:
        self.method: str = method
        self.parametric: bool = parametric
        self.mean_only: bool = bool(mean_only)
        self.reference_batch: Optional[str] = reference_batch
        self.eps: float = float(eps)
        self.covbat_cov_thresh: float = float(covbat_cov_thresh)

        self._batch_levels: pd.Index
        self._grand_mean: pd.Series
        self._pooled_var: pd.Series
        self._gamma_star: FloatArray
        self._delta_star: FloatArray
        self._n_per_batch: Dict[str, int]
        self._reference_batch_idx: Optional[int]
        self._beta_hat_nonbatch: FloatArray
        self._n_batch: int
        self._p_design: int
        self._covbat_pca: PCA
        self._covbat_n_pc: int
        self._batch_levels_pc: pd.Index
        self._pc_gamma_star: FloatArray
        self._pc_delta_star: FloatArray
        
        if not (0.0 < self.covbat_cov_thresh <= 1.0):
            raise ValueError("covbat_cov_thresh must be in (0, 1].")

    @staticmethod
    def _as_series(
        arr: ArrayLike,
        index: pd.Index,
        name: str
    ) -> pd.Series:
        """Convert array-like to categorical Series with validation."""
        if isinstance(arr, pd.Series):
            ser = arr.copy()
        else:
            ser = pd.Series(arr, index=index, name=name)
        if not ser.index.equals(index):
            raise ValueError(f"`{name}` index mismatch with `X`.")
        return ser.astype("category")

    @staticmethod
    def _to_df(
        arr: Optional[ArrayLike],
        index: pd.Index,
        name: str
    ) -> Optional[pd.DataFrame]:
        """Convert array-like to DataFrame."""
        if arr is None:
            return None
        if isinstance(arr, pd.Series):
            arr = arr.to_frame()
        if not isinstance(arr, pd.DataFrame):
            arr = pd.DataFrame(arr, index=index)
        if not arr.index.equals(index):
            raise ValueError(f"`{name}` index mismatch with `X`.")
        return arr

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        *,
        batch: ArrayLike,
        discrete_covariates: Optional[ArrayLike] = None,
        continuous_covariates: Optional[ArrayLike] = None,
    ) -> ComBatModel:
        """Fit the ComBat model."""
        method = self.method.lower()
        if method not in {"johnson", "fortin", "chen"}:
            raise ValueError("method must be 'johnson', 'fortin', or 'chen'.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        idx = X.index
        batch = self._as_series(batch, idx, "batch")

        disc = self._to_df(discrete_covariates, idx, "discrete_covariates")
        cont = self._to_df(continuous_covariates, idx, "continuous_covariates")

        if self.reference_batch is not None and self.reference_batch not in batch.cat.categories:
            raise ValueError(
                f"reference_batch={self.reference_batch!r} not present in the data batches."
                f"{list(batch.cat.categories)}"
            )

        if method == "johnson":
            if disc is not None or cont is not None:
                warnings.warn(
                    "Covariates are ignored when using method='johnson'."
                )
            self._fit_johnson(X, batch)
        elif method == "fortin":
            self._fit_fortin(X, batch, disc, cont)
        elif method == "chen":
            self._fit_chen(X, batch, disc, cont)
        return self

    def _fit_johnson(
        self,
        X: pd.DataFrame,
        batch: pd.Series
    ) -> None:
        """Johnson et al. (2007) ComBat."""
        self._batch_levels = batch.cat.categories
        pooled_var = X.var(axis=0, ddof=1) + self.eps
        grand_mean = X.mean(axis=0)

        Xs = (X - grand_mean) / np.sqrt(pooled_var)

        n_per_batch: Dict[str, int] = {}
        gamma_hat: list[npt.NDArray[np.float64]] = []
        delta_hat: list[npt.NDArray[np.float64]] = []
        
        for lvl in self._batch_levels:
            idx = batch == lvl
            n_b = int(idx.sum())
            if n_b < 2:
                raise ValueError(f"Batch '{lvl}' has <2 samples.")
            n_per_batch[str(lvl)] = n_b
            xb = Xs.loc[idx]
            gamma_hat.append(xb.mean(axis=0).values)
            delta_hat.append(xb.var(axis=0, ddof=1).values + self.eps)

        gamma_hat_arr = np.vstack(gamma_hat)
        delta_hat_arr = np.vstack(delta_hat)

        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat_arr, delta_hat_arr, n_per_batch, parametric=self.parametric
            )
            delta_star = np.ones_like(delta_hat_arr)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat_arr, delta_hat_arr, n_per_batch, parametric=self.parametric
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

    def _fit_fortin(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: Optional[pd.DataFrame],
        cont: Optional[pd.DataFrame],
    ) -> None:
        """Fortin et al. (2018) neuroComBat."""
        self._batch_levels = batch.cat.categories
        n_batch = len(self._batch_levels)
        n_samples = len(X)

        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)
        if self.reference_batch is not None:
            if self.reference_batch not in self._batch_levels:
                raise ValueError(
                    f"reference_batch={self.reference_batch!r} not present in batches."
                    f"{list(self._batch_levels)}"
                )
            batch_dummies.loc[:, self.reference_batch] = 1.0

        parts: list[pd.DataFrame] = [batch_dummies]
        if disc is not None:
            parts.append(
                pd.get_dummies(
                    disc.astype("category"), drop_first=True
                ).astype(float)
            )

        if cont is not None:
            parts.append(cont.astype(float))

        design = pd.concat(parts, axis=1).values
        p_design = design.shape[1]

        X_np = X.values
        beta_hat = la.lstsq(design, X_np, rcond=None)[0]

        beta_hat_batch = beta_hat[:n_batch]
        self._beta_hat_nonbatch = beta_hat[n_batch:]

        n_per_batch = batch.value_counts().sort_index().astype(int).values
        self._n_per_batch = dict(zip(self._batch_levels, n_per_batch))

        if self.reference_batch is not None:
            ref_idx = list(self._batch_levels).index(self.reference_batch)
            grand_mean = beta_hat_batch[ref_idx]
        else:
            grand_mean = (n_per_batch / n_samples) @ beta_hat_batch
            ref_idx = None

        self._grand_mean = pd.Series(grand_mean, index=X.columns)

        if self.reference_batch is not None:
            ref_mask = (batch == self.reference_batch).values
            resid = X_np[ref_mask] - design[ref_mask] @ beta_hat
            denom = int(ref_mask.sum())
        else:
            resid = X_np - design @ beta_hat
            denom = n_samples
        var_pooled = (resid ** 2).sum(axis=0) / denom + self.eps
        self._pooled_var = pd.Series(var_pooled, index=X.columns)

        stand_mean = grand_mean + design[:, n_batch:] @ self._beta_hat_nonbatch
        Xs = (X_np - stand_mean) / np.sqrt(var_pooled)

        gamma_hat = np.vstack(
            [Xs[batch == lvl].mean(axis=0) for lvl in self._batch_levels]
        )
        delta_hat = np.vstack(
            [Xs[batch == lvl].var(axis=0, ddof=1) + self.eps
             for lvl in self._batch_levels]
        )

        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat, delta_hat, n_per_batch,
                parametric = self.parametric
            )
            delta_star = np.ones_like(delta_hat)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat, delta_hat, n_per_batch,
                parametric = self.parametric
            )

        if ref_idx is not None:
            gamma_star[ref_idx] = 0.0
            if not self.mean_only:
                delta_star[ref_idx] = 1.0
        self._reference_batch_idx = ref_idx

        self._gamma_star = gamma_star
        self._delta_star = delta_star
        self._n_batch  = n_batch
        self._p_design = p_design
    
    def _fit_chen(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: Optional[pd.DataFrame],
        cont: Optional[pd.DataFrame],
    ) -> None:
        """Chen et al. (2022) CovBat."""
        self._fit_fortin(X, batch, disc, cont)
        X_meanvar_adj = self._transform_fortin(X, batch, disc, cont)
        X_centered = X_meanvar_adj - X_meanvar_adj.mean(axis=0)
        pca = PCA(svd_solver="full", whiten=False).fit(X_centered)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        n_pc = int(np.searchsorted(cumulative, self.covbat_cov_thresh) + 1)
        self._covbat_pca = pca
        self._covbat_n_pc = n_pc

        scores = pca.transform(X_centered)[:, :n_pc]
        scores_df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(n_pc)])
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

        if self.mean_only:
            gamma_star = self._shrink_gamma(
                gamma_hat_arr, delta_hat_arr, n_per_batch, parametric=self.parametric
            )
            delta_star = np.ones_like(delta_hat_arr)
        else:
            gamma_star, delta_star = self._shrink_gamma_delta(
                gamma_hat_arr, delta_hat_arr, n_per_batch, parametric=self.parametric
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

    def _shrink_gamma_delta(
        self,
        gamma_hat: FloatArray,
        delta_hat: FloatArray,
        n_per_batch: Union[Dict[str, int], FloatArray],
        *,
        parametric: bool,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> Tuple[FloatArray, FloatArray]:
        """Empirical Bayes shrinkage estimation."""
        if parametric:
            gamma_bar = gamma_hat.mean(axis=0)
            t2 = gamma_hat.var(axis=0, ddof=1)
            a_prior = (delta_hat.mean(axis=0) ** 2) / delta_hat.var(axis=0, ddof=1) + 2
            b_prior = delta_hat.mean(axis=0) * (a_prior - 1)

            B, p = gamma_hat.shape
            gamma_star = np.empty_like(gamma_hat)
            delta_star = np.empty_like(delta_hat)
            n_vec = np.array(list(n_per_batch.values())) if isinstance(n_per_batch, dict) else n_per_batch

            for i in range(B):
                n_i = n_vec[i]
                g, d = gamma_hat[i], delta_hat[i]
                gamma_post_var = 1.0 / (n_i / d + 1.0 / t2)
                gamma_star[i] = gamma_post_var * (n_i * g / d + gamma_bar / t2)

                a_post = a_prior + n_i / 2.0
                b_post = b_prior + 0.5 * n_i * d
                delta_star[i] = b_post / (a_post - 1)
            return gamma_star, delta_star

        else:
            B, p = gamma_hat.shape
            n_vec = np.array(list(n_per_batch.values())) if isinstance(n_per_batch, dict) else n_per_batch
            gamma_bar = gamma_hat.mean(axis=0)
            t2 = gamma_hat.var(axis=0, ddof=1)

            def postmean(
                g_hat: FloatArray,
                g_bar: FloatArray,
                n: float,
                d_star: FloatArray,
                t2_: FloatArray
            ) -> FloatArray:
                return (t2_ * n * g_hat + d_star * g_bar) / (t2_ * n + d_star)

            def postvar(
                sum2: FloatArray,
                n: float,
                a: FloatArray,
                b: FloatArray
            ) -> FloatArray:
                return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

            def aprior(delta: FloatArray) -> FloatArray:
                m, s2 = delta.mean(), delta.var()
                s2 = max(s2, self.eps)
                return (2 * s2 + m ** 2) / s2

            def bprior(delta: FloatArray) -> FloatArray:
                m, s2 = delta.mean(), delta.var()
                s2 = max(s2, self.eps)
                return (m * s2 + m ** 3) / s2

            gamma_star = np.empty_like(gamma_hat)
            delta_star = np.empty_like(delta_hat)

            for i in range(B):
                n_i = n_vec[i]
                g_hat_i = gamma_hat[i]
                d_hat_i = delta_hat[i]
                a_i = aprior(d_hat_i)
                b_i = bprior(d_hat_i)

                g_new, d_new = g_hat_i.copy(), d_hat_i.copy()
                for _ in range(max_iter):
                    g_prev, d_prev = g_new, d_new
                    g_new = postmean(g_hat_i, gamma_bar, n_i, d_prev, t2)
                    sum2 = (n_i - 1) * d_hat_i + n_i * (g_hat_i - g_new) ** 2
                    d_new = postvar(sum2, n_i, a_i, b_i)
                    if np.max(np.abs(g_new - g_prev) / (np.abs(g_prev) + self.eps)) < tol and (
                        self.mean_only or np.max(np.abs(d_new - d_prev) / (np.abs(d_prev) + self.eps)) < tol
                    ):
                        break
                gamma_star[i] = g_new
                delta_star[i] = 1.0 if self.mean_only else d_new
            return gamma_star, delta_star

    def _shrink_gamma(
        self,
        gamma_hat: FloatArray,
        delta_hat: FloatArray,
        n_per_batch: Union[Dict[str, int], FloatArray],
        *,
        parametric: bool,
    ) -> FloatArray:
        """Convenience wrapper that returns only γ⋆ (for *mean-only* mode)."""
        gamma, _ = self._shrink_gamma_delta(gamma_hat, delta_hat, n_per_batch, parametric=parametric)
        return gamma

    def transform(
        self,
        X: ArrayLike,
        *,
        batch: ArrayLike,
        discrete_covariates: Optional[ArrayLike] = None,
        continuous_covariates: Optional[ArrayLike] = None,
    ) -> pd.DataFrame:
        """Transform the data using fitted ComBat parameters."""
        check_is_fitted(self, ["_gamma_star"])
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
        else:
            raise ValueError(f"Unknown method: {method}.")

    def _transform_johnson(
        self,
        X: pd.DataFrame,
        batch: pd.Series
    ) -> pd.DataFrame:
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
            if self.mean_only:
                Xb = Xs.loc[idx] - g
            else:
                Xb = (Xs.loc[idx] - g) / np.sqrt(d)
            X_adj.loc[idx] = (Xb * np.sqrt(pooled) + grand).values
        return X_adj

    def _transform_fortin(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: Optional[pd.DataFrame],
        cont: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Fortin transform implementation."""
        batch_dummies = pd.get_dummies(batch, drop_first=False).astype(float)[self._batch_levels]
        if self.reference_batch is not None:
            batch_dummies.loc[:, self.reference_batch] = 1.0

        parts = [batch_dummies]
        if disc is not None:
            parts.append(
                pd.get_dummies(
                    disc.astype("category"), drop_first=True
                ).astype(float)
            )
        if cont is not None:
            parts.append(cont.astype(float))

        design = pd.concat(parts, axis=1).values

        X_np = X.values
        stand_mu = (
            self._grand_mean.values +
            design[:, self._n_batch:] @ self._beta_hat_nonbatch
        )
        Xs = (X_np - stand_mu) / np.sqrt(self._pooled_var.values)

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

        X_adj = (
            Xs * np.sqrt(self._pooled_var.values) +
            stand_mu
        )
        return pd.DataFrame(X_adj, index=X.index, columns=X.columns, dtype=float)
    
    def _transform_chen(
        self,
        X: pd.DataFrame,
        batch: pd.Series,
        disc: Optional[pd.DataFrame],
        cont: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Chen transform implementation."""
        X_meanvar_adj = self._transform_fortin(X, batch, disc, cont)
        X_centered = X_meanvar_adj - self._covbat_pca.mean_
        scores = self._covbat_pca.transform(X_centered.values)
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

        X_recon = self._covbat_pca.inverse_transform(scores_adj) + self._covbat_pca.mean_
        return pd.DataFrame(X_recon, index=X.index, columns=X.columns)


class ComBat(BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper around `ComBatModel`.

    Stores batch (and optional covariates) passed at construction and
    appropriately uses them for separate `fit` and `transform`.
    """

    def __init__(
        self,
        batch: ArrayLike,
        *,
        discrete_covariates: Optional[ArrayLike] = None,
        continuous_covariates: Optional[ArrayLike] = None,
        method: str = "johnson",
        parametric: bool = True,
        mean_only: bool = False,
        reference_batch: Optional[str] = None,
        eps: float = 1e-8,
        covbat_cov_thresh: float = 0.9,
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
        self._model = ComBatModel(
            method=method,
            parametric=parametric,
            mean_only=mean_only,
            reference_batch=reference_batch,
            eps=eps,
            covbat_cov_thresh=covbat_cov_thresh,
        )

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None
    ) -> "ComBat":
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
    def _subset(
        obj: Optional[ArrayLike],
        idx: pd.Index
    ) -> Optional[Union[pd.DataFrame, pd.Series]]:
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

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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import levene, spearmanr, chi2
from scipy.spatial.distance import pdist
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Literal, Optional, Union, Dict, Tuple, Any, List
import numpy.typing as npt
import warnings
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ArrayLike = Union[pd.DataFrame, pd.Series, npt.NDArray[Any]]
FloatArray = npt.NDArray[np.float64]

def _compute_pca_embedding(
    X_before: np.ndarray,
    X_after: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Compute PCA embeddings for both datasets.

    Fits PCA on X_before and applies to both datasets.

    Parameters
    ----------
    X_before : np.ndarray
        Original data before correction.
    X_after : np.ndarray
        Corrected data.
    n_components : int
        Number of PCA components.

    Returns
    -------
    X_before_pca : np.ndarray
        PCA-transformed original data.
    X_after_pca : np.ndarray
        PCA-transformed corrected data.
    pca : PCA
        Fitted PCA model.
    """
    n_components = min(n_components, X_before.shape[1], X_before.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    X_before_pca = pca.fit_transform(X_before)
    X_after_pca = pca.transform(X_after)
    return X_before_pca, X_after_pca, pca


def _silhouette_batch(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute silhouette coefficient using batch as cluster labels.

    Lower values after correction indicate better batch mixing.
    Range: [-1, 1], where -1 = batch mixing, 1 = batch separation.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.

    Returns
    -------
    float
        Silhouette coefficient.
    """
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0
    try:
        return silhouette_score(X, batch_labels, metric='euclidean')
    except Exception:
        return 0.0


def _davies_bouldin_batch(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin index using batch labels.

    Lower values indicate better batch mixing.
    Range: [0, inf), 0 = perfect batch overlap.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.

    Returns
    -------
    float
        Davies-Bouldin index.
    """
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0
    try:
        return davies_bouldin_score(X, batch_labels)
    except Exception:
        return 0.0


def _kbet_score(
    X: np.ndarray,
    batch_labels: np.ndarray,
    k0: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute kBET (k-nearest neighbor Batch Effect Test) acceptance rate.

    Tests if local batch proportions match global batch proportions.
    Higher acceptance rate = better batch mixing.

    Reference: Buttner et al. (2019) Nature Methods

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.
    k0 : int
        Neighborhood size.
    alpha : float
        Significance level for chi-squared test.

    Returns
    -------
    acceptance_rate : float
        Fraction of samples where H0 (uniform mixing) is accepted.
    mean_stat : float
        Mean chi-squared statistic across samples.
    """
    n_samples = X.shape[0]
    unique_batches, batch_counts = np.unique(batch_labels, return_counts=True)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 1.0, 0.0

    global_freq = batch_counts / n_samples
    k0 = min(k0, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k0 + 1, algorithm='auto')
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    chi2_stats = []
    p_values = []
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}

    for i in range(n_samples):
        neighbors = indices[i, 1:k0+1]
        neighbor_batches = batch_labels[neighbors]

        observed = np.zeros(n_batches)
        for nb in neighbor_batches:
            observed[batch_to_idx[nb]] += 1

        expected = global_freq * k0

        mask = expected > 0
        if mask.sum() < 2:
            continue

        stat = np.sum((observed[mask] - expected[mask])**2 / expected[mask])
        df = max(1, mask.sum() - 1)
        p_val = 1 - chi2.cdf(stat, df)

        chi2_stats.append(stat)
        p_values.append(p_val)

    if len(p_values) == 0:
        return 1.0, 0.0

    acceptance_rate = np.mean(np.array(p_values) > alpha)
    mean_stat = np.mean(chi2_stats)

    return acceptance_rate, mean_stat


def _find_sigma(distances: np.ndarray, target_perplexity: float, tol: float = 1e-5) -> float:
    """
    Binary search for sigma to achieve target perplexity.

    Used in LISI computation.

    Parameters
    ----------
    distances : np.ndarray
        Distances to neighbors.
    target_perplexity : float
        Target perplexity value.
    tol : float
        Tolerance for convergence.

    Returns
    -------
    float
        Sigma value.
    """
    target_H = np.log2(target_perplexity + 1e-10)

    sigma_min, sigma_max = 1e-10, 1e10
    sigma = 1.0

    for _ in range(50):
        P = np.exp(-distances**2 / (2 * sigma**2 + 1e-10))
        P_sum = P.sum()
        if P_sum < 1e-10:
            sigma = (sigma + sigma_max) / 2
            continue
        P = P / P_sum
        P = np.clip(P, 1e-10, 1.0)
        H = -np.sum(P * np.log2(P))

        if abs(H - target_H) < tol:
            break
        elif H < target_H:
            sigma_min = sigma
        else:
            sigma_max = sigma
        sigma = (sigma_min + sigma_max) / 2

    return sigma


def _lisi_score(
    X: np.ndarray,
    batch_labels: np.ndarray,
    perplexity: int = 30,
) -> float:
    """
    Compute mean Local Inverse Simpson's Index (LISI).

    Range: [1, n_batches], where n_batches = perfect mixing.
    Higher = better batch mixing.

    Reference: Korsunsky et al. (2019) Nature Methods (Harmony paper)

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.
    perplexity : int
        Perplexity for Gaussian kernel.

    Returns
    -------
    float
        Mean LISI score.
    """
    n_samples = X.shape[0]
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}

    if n_batches < 2:
        return 1.0

    k = min(3 * perplexity, n_samples - 1)

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    lisi_values = []

    for i in range(n_samples):
        sigma = _find_sigma(distances[i], perplexity)

        P = np.exp(-distances[i]**2 / (2 * sigma**2 + 1e-10))
        P_sum = P.sum()
        if P_sum < 1e-10:
            lisi_values.append(1.0)
            continue
        P = P / P_sum

        neighbor_batches = batch_labels[indices[i]]
        batch_probs = np.zeros(n_batches)
        for j, nb in enumerate(neighbor_batches):
            batch_probs[batch_to_idx[nb]] += P[j]

        simpson = np.sum(batch_probs**2)
        if simpson < 1e-10:
            lisi = n_batches
        else:
            lisi = 1.0 / simpson
        lisi_values.append(lisi)

    return np.mean(lisi_values)


def _variance_ratio(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute between-batch to within-batch variance ratio.

    Similar to F-statistic in one-way ANOVA.
    Lower ratio after correction = better batch effect removal.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.

    Returns
    -------
    float
        Variance ratio (between/within).
    """
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)
    n_samples = X.shape[0]

    if n_batches < 2:
        return 0.0

    grand_mean = np.mean(X, axis=0)

    between_var = 0.0
    within_var = 0.0

    for batch in unique_batches:
        mask = batch_labels == batch
        n_b = np.sum(mask)
        X_batch = X[mask]
        batch_mean = np.mean(X_batch, axis=0)

        between_var += n_b * np.sum((batch_mean - grand_mean)**2)
        within_var += np.sum((X_batch - batch_mean)**2)

    between_var /= (n_batches - 1)
    within_var /= (n_samples - n_batches)

    if within_var < 1e-10:
        return 0.0

    return between_var / within_var


def _knn_preservation(
    X_before: np.ndarray,
    X_after: np.ndarray,
    k_values: List[int],
    n_jobs: int = 1,
) -> Dict[int, float]:
    """
    Compute fraction of k-nearest neighbors preserved after correction.

    Range: [0, 1], where 1 = perfect preservation.
    Higher = better biological structure preservation.

    Parameters
    ----------
    X_before : np.ndarray
        Original data.
    X_after : np.ndarray
        Corrected data.
    k_values : list of int
        Values of k for k-NN.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    dict
        Mapping from k to preservation fraction.
    """
    results = {}
    max_k = max(k_values)
    max_k = min(max_k, X_before.shape[0] - 1)

    nn_before = NearestNeighbors(n_neighbors=max_k + 1, algorithm='auto', n_jobs=n_jobs)
    nn_before.fit(X_before)
    _, indices_before = nn_before.kneighbors(X_before)

    nn_after = NearestNeighbors(n_neighbors=max_k + 1, algorithm='auto', n_jobs=n_jobs)
    nn_after.fit(X_after)
    _, indices_after = nn_after.kneighbors(X_after)

    for k in k_values:
        if k > max_k:
            results[k] = 0.0
            continue

        overlaps = []
        for i in range(X_before.shape[0]):
            neighbors_before = set(indices_before[i, 1:k+1])
            neighbors_after = set(indices_after[i, 1:k+1])
            overlap = len(neighbors_before & neighbors_after) / k
            overlaps.append(overlap)

        results[k] = np.mean(overlaps)

    return results


def _pairwise_distance_correlation(
    X_before: np.ndarray,
    X_after: np.ndarray,
    subsample: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Compute Spearman correlation of pairwise distances.

    Range: [-1, 1], where 1 = perfect rank preservation.
    Higher = better relative relationship preservation.

    Parameters
    ----------
    X_before : np.ndarray
        Original data.
    X_after : np.ndarray
        Corrected data.
    subsample : int
        Maximum samples to use (for efficiency).
    random_state : int
        Random seed for subsampling.

    Returns
    -------
    float
        Spearman correlation coefficient.
    """
    n_samples = X_before.shape[0]

    if n_samples > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, subsample, replace=False)
        X_before = X_before[idx]
        X_after = X_after[idx]

    dist_before = pdist(X_before, metric='euclidean')
    dist_after = pdist(X_after, metric='euclidean')

    if len(dist_before) == 0:
        return 1.0

    corr, _ = spearmanr(dist_before, dist_after)

    if np.isnan(corr):
        return 1.0

    return corr


def _mean_centroid_distance(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute mean pairwise Euclidean distance between batch centroids.

    Lower after correction = better batch alignment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.

    Returns
    -------
    float
        Mean pairwise distance between centroids.
    """
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        return 0.0

    centroids = []
    for batch in unique_batches:
        mask = batch_labels == batch
        centroid = np.mean(X[mask], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    distances = pdist(centroids, metric='euclidean')

    return np.mean(distances)


def _levene_median_statistic(X: np.ndarray, batch_labels: np.ndarray) -> float:
    """
    Compute median Levene test statistic across features.

    Lower statistic = more homogeneous variances across batches.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    batch_labels : np.ndarray
        Batch labels for each sample.

    Returns
    -------
    float
        Median Levene test statistic.
    """
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0

    levene_stats = []
    for j in range(X.shape[1]):
        groups = [X[batch_labels == b, j] for b in unique_batches]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            stat, _ = levene(*groups, center='median')
            if not np.isnan(stat):
                levene_stats.append(stat)
        except Exception:
            continue

    if len(levene_stats) == 0:
        return 0.0

    return np.median(levene_stats)


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
    covbat_cov_thresh : float or int, default=0.9
        CovBat: cumulative variance threshold (0, 1] to retain PCs, or
        integer >= 1 specifying the number of components directly.
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
        covbat_cov_thresh: Union[float, int] = 0.9,
    ) -> None:
        self.method: str = method
        self.parametric: bool = parametric
        self.mean_only: bool = bool(mean_only)
        self.reference_batch: Optional[str] = reference_batch
        self.eps: float = float(eps)
        self.covbat_cov_thresh: Union[float, int] = covbat_cov_thresh

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

        # Validate covbat_cov_thresh
        if isinstance(self.covbat_cov_thresh, float):
            if not (0.0 < self.covbat_cov_thresh <= 1.0):
                raise ValueError("covbat_cov_thresh must be in (0, 1] when float.")
        elif isinstance(self.covbat_cov_thresh, int):
            if self.covbat_cov_thresh < 1:
                raise ValueError("covbat_cov_thresh must be >= 1 when int.")
        else:
            raise TypeError("covbat_cov_thresh must be float or int.")

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

        # Determine number of components based on threshold type
        if isinstance(self.covbat_cov_thresh, int):
            n_pc = min(self.covbat_cov_thresh, len(pca.explained_variance_ratio_))
        else:
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
        if not hasattr(self, "_gamma_star"):
            raise ValueError("This ComBatModel instance is not fitted yet. Call 'fit' before 'transform'.")
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
        scores = self._covbat_pca.transform(X_centered)
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
        covbat_cov_thresh: Union[float, int] = 0.9,
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
        self._fitted_batch = batch_vec
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

    @property
    def metrics_(self) -> Optional[Dict[str, Any]]:
        """Return cached metrics from last fit_transform with compute_metrics=True.

        Returns
        -------
        dict or None
            Cached metrics dictionary, or None if no metrics have been computed.
        """
        return getattr(self, '_metrics_cache', None)

    def compute_batch_metrics(
        self,
        X: ArrayLike,
        batch: Optional[ArrayLike] = None,
        *,
        pca_components: Optional[int] = None,
        k_neighbors: List[int] = [5, 10, 50],
        kbet_k0: Optional[int] = None,
        lisi_perplexity: int = 30,
        n_jobs: int = 1,
    ) -> Dict[str, Any]:
        """
        Compute batch effect metrics before and after ComBat correction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to evaluate.
        batch : array-like of shape (n_samples,), optional
            Batch labels. If None, uses the batch stored at construction.
        pca_components : int, optional
            Number of PCA components for dimensionality reduction before
            computing metrics. If None (default), metrics are computed in
            the original feature space. Must be less than min(n_samples, n_features).
        k_neighbors : list of int, default=[5, 10, 50]
            Values of k for k-NN preservation metric.
        kbet_k0 : int, optional
            Neighborhood size for kBET. Default is 10% of samples.
        lisi_perplexity : int, default=30
            Perplexity for LISI computation.
        n_jobs : int, default=1
            Number of parallel jobs for neighbor computations.

        Returns
        -------
        metrics : dict
            Dictionary with structure:
            {
                'batch_effect': {
                    'silhouette': {'before': float, 'after': float},
                    'davies_bouldin': {...},
                    'kbet': {...},
                    'lisi': {..., 'max_value': n_batches},
                    'variance_ratio': {...},
                },
                'preservation': {
                    'knn': {k: fraction for k in k_neighbors},
                    'distance_correlation': float,
                },
                'alignment': {
                    'centroid_distance': {...},
                    'levene_statistic': {...},
                },
            }

        Raises
        ------
        ValueError
            If the model is not fitted or if pca_components is invalid.
        """
        if not hasattr(self._model, "_gamma_star"):
            raise ValueError(
                "This ComBat instance is not fitted yet. "
                "Call 'fit' before 'compute_batch_metrics'."
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        idx = X.index

        if batch is None:
            batch_vec = self._subset(self.batch, idx)
        else:
            if isinstance(batch, (pd.Series, pd.DataFrame)):
                batch_vec = batch.loc[idx] if hasattr(batch, 'loc') else batch
            elif isinstance(batch, np.ndarray):
                batch_vec = pd.Series(batch, index=idx)
            else:
                batch_vec = pd.Series(batch, index=idx)

        batch_labels = np.array(batch_vec)

        X_before = X.values
        X_after = self.transform(X).values

        n_samples, n_features = X_before.shape
        if kbet_k0 is None:
            kbet_k0 = max(10, int(0.10 * n_samples))

        # Validate and apply PCA if requested
        if pca_components is not None:
            max_components = min(n_samples, n_features)
            if pca_components >= max_components:
                raise ValueError(
                    f"pca_components={pca_components} must be less than "
                    f"min(n_samples, n_features)={max_components}."
                )
            X_before_pca, X_after_pca, _ = _compute_pca_embedding(
                X_before, X_after, pca_components
            )
        else:
            X_before_pca = X_before
            X_after_pca = X_after

        silhouette_before = _silhouette_batch(X_before_pca, batch_labels)
        silhouette_after = _silhouette_batch(X_after_pca, batch_labels)

        db_before = _davies_bouldin_batch(X_before_pca, batch_labels)
        db_after = _davies_bouldin_batch(X_after_pca, batch_labels)

        kbet_before, _ = _kbet_score(X_before_pca, batch_labels, kbet_k0)
        kbet_after, _ = _kbet_score(X_after_pca, batch_labels, kbet_k0)

        lisi_before = _lisi_score(X_before_pca, batch_labels, lisi_perplexity)
        lisi_after = _lisi_score(X_after_pca, batch_labels, lisi_perplexity)

        var_ratio_before = _variance_ratio(X_before_pca, batch_labels)
        var_ratio_after = _variance_ratio(X_after_pca, batch_labels)

        knn_results = _knn_preservation(X_before_pca, X_after_pca, k_neighbors, n_jobs)
        dist_corr = _pairwise_distance_correlation(X_before_pca, X_after_pca)

        centroid_before = _mean_centroid_distance(X_before_pca, batch_labels)
        centroid_after = _mean_centroid_distance(X_after_pca, batch_labels)

        levene_before = _levene_median_statistic(X_before, batch_labels)
        levene_after = _levene_median_statistic(X_after, batch_labels)

        n_batches = len(np.unique(batch_labels))

        metrics = {
            'batch_effect': {
                'silhouette': {
                    'before': silhouette_before,
                    'after': silhouette_after,
                },
                'davies_bouldin': {
                    'before': db_before,
                    'after': db_after,
                },
                'kbet': {
                    'before': kbet_before,
                    'after': kbet_after,
                },
                'lisi': {
                    'before': lisi_before,
                    'after': lisi_after,
                    'max_value': n_batches,
                },
                'variance_ratio': {
                    'before': var_ratio_before,
                    'after': var_ratio_after,
                },
            },
            'preservation': {
                'knn': knn_results,
                'distance_correlation': dist_corr,
            },
            'alignment': {
                'centroid_distance': {
                    'before': centroid_before,
                    'after': centroid_after,
                },
                'levene_statistic': {
                    'before': levene_before,
                    'after': levene_after,
                },
            },
        }

        return metrics

    def fit_transform(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None
    ) -> pd.DataFrame:
        """
        Fit and transform the data, optionally computing metrics.

        If compute_metrics=True was set at construction, batch effect
        metrics are computed and cached in metrics_ property.

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

    def plot_transformation(
            self,
            X: ArrayLike, *, 
            reduction_method: Literal['pca', 'tsne', 'umap'] = 'pca', 
            n_components: Literal[2, 3] = 2,
            plot_type: Literal['static', 'interactive'] = 'static',
            figsize: Tuple[int, int] = (12, 5),
            alpha: float = 0.7,
            point_size: int = 50,
            cmap: str = 'Set1',
            title: Optional[str] = None,
            show_legend: bool = True,
            return_embeddings: bool = False,
            **reduction_kwargs) -> Union[Any, Tuple[Any, Dict[str, FloatArray]]]:
        """
        Visualize the ComBat transformation effect using dimensionality reduction.

        It shows a before/after comparison of data transformed by `ComBat` using
        PCA, t-SNE, or UMAP to reduce dimensions for visualization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform and visualize.

        reduction_method : {`'pca'`, `'tsne'`, `'umap'`}, default=`'pca'`
            Dimensionality reduction method.

        n_components : {2, 3}, default=2
            Number of components for dimensionality reduction.

        plot_type : {`'static'`, `'interactive'`}, default=`'static'`
            Visualization type:
            - `'static'`: matplotlib plots (can be saved as images)
            - `'interactive'`: plotly plots (explorable, requires plotly)

        return_embeddings : bool, default=False
            If `True`, return embeddings along with the plot.
            
        **reduction_kwargs : dict
            Additional parameters for reduction methods.
            
        Returns
        -------
        fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
            The figure object containing the plots.
            
        embeddings : dict, optional
            If `return_embeddings=True`, dictionary with:
            - `'original'`: embedding of original data
            - `'transformed'`: embedding of ComBat-transformed data
        """
        if not hasattr(self._model, "_gamma_star"):
            raise ValueError("This ComBat instance is not fitted yet. Call 'fit' before 'plot_transformation'.")

        if n_components not in [2, 3]:
            raise ValueError(f"n_components must be 2 or 3, got {n_components}")
        if reduction_method not in ['pca', 'tsne', 'umap']:
            raise ValueError(f"reduction_method must be 'pca', 'tsne', or 'umap', got '{reduction_method}'")
        if plot_type not in ['static', 'interactive']:
            raise ValueError(f"plot_type must be 'static' or 'interactive', got '{plot_type}'")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        idx = X.index
        batch_vec = self._subset(self.batch, idx)
        if batch_vec is None:
            raise ValueError("Batch information is required for visualization")

        X_transformed = self.transform(X)

        X_np = X.values
        X_trans_np = X_transformed.values

        if reduction_method == 'pca':
            reducer_orig = PCA(n_components=n_components, **reduction_kwargs)
            reducer_trans = PCA(n_components=n_components, **reduction_kwargs)
        elif reduction_method == 'tsne':
            tsne_params = {'perplexity': 30, 'max_iter': 1000, 'random_state': 42}
            tsne_params.update(reduction_kwargs)
            reducer_orig = TSNE(n_components=n_components, **tsne_params)
            reducer_trans = TSNE(n_components=n_components, **tsne_params)
        else:
            umap_params = {'random_state': 42}
            umap_params.update(reduction_kwargs)
            reducer_orig = umap.UMAP(n_components=n_components, **umap_params)
            reducer_trans = umap.UMAP(n_components=n_components, **umap_params)

        X_embedded_orig = reducer_orig.fit_transform(X_np)
        X_embedded_trans = reducer_trans.fit_transform(X_trans_np)

        if plot_type == 'static':
            fig = self._create_static_plot(
                X_embedded_orig, X_embedded_trans, batch_vec, 
                reduction_method, n_components, figsize, alpha, 
                point_size, cmap, title, show_legend
            )
        else:
            fig = self._create_interactive_plot(
                X_embedded_orig, X_embedded_trans, batch_vec,
                reduction_method, n_components, cmap, title, show_legend
            )

        if return_embeddings:
            embeddings = {
                'original': X_embedded_orig,
                'transformed': X_embedded_trans
            }
            return fig, embeddings
        else:
            return fig

    def _create_static_plot(
            self,
            X_orig: FloatArray,
            X_trans: FloatArray, 
            batch_labels: pd.Series,
            method: str,
            n_components: int,
            figsize: Tuple[int, int],
            alpha: float,
            point_size: int, 
            cmap: str,
            title: Optional[str],
            show_legend: bool) -> Any:
        """Create static plots using matplotlib."""

        fig = plt.figure(figsize=figsize)

        unique_batches = batch_labels.drop_duplicates()
        n_batches = len(unique_batches)

        if n_batches <= 10:
            colors = matplotlib.colormaps.get_cmap(cmap)(np.linspace(0, 1, n_batches))
        else:
            colors = matplotlib.colormaps.get_cmap('tab20')(np.linspace(0, 1, n_batches))

        if n_components == 2:
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
        else:
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            if n_components == 2:
                ax1.scatter(
                    X_orig[mask, 0], X_orig[mask, 1], 
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha, 
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )
            else:
                ax1.scatter(
                    X_orig[mask, 0], X_orig[mask, 1], X_orig[mask, 2],
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha, 
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )

        ax1.set_title(f'Before ComBat correction\n({method.upper()})')
        ax1.set_xlabel(f'{method.upper()}1')
        ax1.set_ylabel(f'{method.upper()}2')
        if n_components == 3:
            ax1.set_zlabel(f'{method.upper()}3')

        for i, batch in enumerate(unique_batches):
            mask = batch_labels == batch
            if n_components == 2:
                ax2.scatter(
                    X_trans[mask, 0], X_trans[mask, 1], 
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha, 
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )
            else:
                ax2.scatter(
                    X_trans[mask, 0], X_trans[mask, 1], X_trans[mask, 2],
                    c=[colors[i]],
                    s=point_size,
                    alpha=alpha, 
                    label=f'Batch {batch}',
                    edgecolors='black',
                    linewidth=0.5
                )

        ax2.set_title(f'After ComBat correction\n({method.upper()})')
        ax2.set_xlabel(f'{method.upper()}1')
        ax2.set_ylabel(f'{method.upper()}2')
        if n_components == 3:
            ax2.set_zlabel(f'{method.upper()}3')

        if show_legend and n_batches <= 20:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if title is None:
            title = f'ComBat correction effect visualized with {method.upper()}'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def _create_interactive_plot(
            self,
            X_orig: FloatArray,
            X_trans: FloatArray,
            batch_labels: pd.Series,
            method: str,
            n_components: int,
            cmap: str,
            title: Optional[str],
            show_legend: bool) -> Any:
        """Create interactive plots using plotly."""
        if n_components == 2:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    f'Before ComBat correction ({method.upper()})',
                    f'After ComBat correction ({method.upper()})'
                )
            )
        else:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
                subplot_titles=(
                    f'Before ComBat correction ({method.upper()})',
                    f'After ComBat correction ({method.upper()})'
                )
            )

        unique_batches = batch_labels.drop_duplicates()

        n_batches = len(unique_batches)
        cmap_func = matplotlib.colormaps.get_cmap(cmap)
        color_list = [mcolors.to_hex(cmap_func(i / max(n_batches - 1, 1))) for i in range(n_batches)]

        batch_to_color = dict(zip(unique_batches, color_list))

        for batch in unique_batches:
            mask = batch_labels == batch

            if n_components == 2:
                fig.add_trace(
                    go.Scatter(
                        x=X_orig[mask, 0], y=X_orig[mask, 1],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=8,
                            color=batch_to_color[batch],
                            line=dict(width=1, color='black')
                        ),
                        showlegend=False),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=X_trans[mask, 0], y=X_trans[mask, 1],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=8,
                            color=batch_to_color[batch],
                            line=dict(width=1, color='black')
                        ),
                        showlegend=show_legend),
                    row=1, col=2
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=X_orig[mask, 0], y=X_orig[mask, 1], z=X_orig[mask, 2],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=5,
                            color=batch_to_color[batch],
                            line=dict(width=0.5, color='black')
                        ),
                        showlegend=False),
                    row=1, col=1
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=X_trans[mask, 0], y=X_trans[mask, 1], z=X_trans[mask, 2],
                        mode='markers',
                        name=f'Batch {batch}',
                        marker=dict(
                            size=5,
                            color=batch_to_color[batch],
                            line=dict(width=0.5, color='black')
                        ),
                        showlegend=show_legend),
                    row=1, col=2
                )

        if title is None:
            title = f'ComBat correction effect visualized with {method.upper()}'

        fig.update_layout(
            title=title,
            title_font_size=16,
            height=600,
            showlegend=show_legend,
            hovermode='closest'
        )

        axis_labels = [f'{method.upper()}{i+1}' for i in range(n_components)]

        if n_components == 2:
            fig.update_xaxes(title_text=axis_labels[0])
            fig.update_yaxes(title_text=axis_labels[1])
        else:
            fig.update_scenes(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2]
            )

        return fig

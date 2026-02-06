"""Batch effect metrics and diagnostics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import chi2, levene, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .core import ArrayLike


def _compute_pca_embedding(
    X_before: np.ndarray,
    X_after: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, PCA]:
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
        return silhouette_score(X, batch_labels, metric="euclidean")
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
) -> tuple[float, float]:
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

    nn = NearestNeighbors(n_neighbors=k0 + 1, algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)

    chi2_stats = []
    p_values = []
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}

    for i in range(n_samples):
        neighbors = indices[i, 1 : k0 + 1]
        neighbor_batches = batch_labels[neighbors]

        observed = np.zeros(n_batches)
        for nb in neighbor_batches:
            observed[batch_to_idx[nb]] += 1

        expected = global_freq * k0

        mask = expected > 0
        if mask.sum() < 2:
            continue

        stat = np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
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
        P = np.exp(-(distances**2) / (2 * sigma**2 + 1e-10))
        P_sum = P.sum()
        if P_sum < 1e-10:
            sigma = (sigma + sigma_max) / 2
            continue
        P = P / P_sum
        P = np.clip(P, 1e-10, 1.0)
        H = -np.sum(P * np.log2(P))

        if abs(H - target_H) < tol:
            break
        elif target_H > H:
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

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    distances = distances[:, 1:]
    indices = indices[:, 1:]

    lisi_values = []

    for i in range(n_samples):
        sigma = _find_sigma(distances[i], perplexity)

        P = np.exp(-(distances[i] ** 2) / (2 * sigma**2 + 1e-10))
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
        lisi = n_batches if simpson < 1e-10 else 1.0 / simpson
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

        between_var += n_b * np.sum((batch_mean - grand_mean) ** 2)
        within_var += np.sum((X_batch - batch_mean) ** 2)

    between_var /= n_batches - 1
    within_var /= n_samples - n_batches

    if within_var < 1e-10:
        return 0.0

    return between_var / within_var


def _knn_preservation(
    X_before: np.ndarray,
    X_after: np.ndarray,
    k_values: list[int],
    n_jobs: int = 1,
) -> dict[int, float]:
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

    nn_before = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", n_jobs=n_jobs)
    nn_before.fit(X_before)
    _, indices_before = nn_before.kneighbors(X_before)

    nn_after = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto", n_jobs=n_jobs)
    nn_after.fit(X_after)
    _, indices_after = nn_after.kneighbors(X_after)

    for k in k_values:
        if k > max_k:
            results[k] = 0.0
            continue

        overlaps = []
        for i in range(X_before.shape[0]):
            neighbors_before = set(indices_before[i, 1 : k + 1])
            neighbors_after = set(indices_after[i, 1 : k + 1])
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

    dist_before = pdist(X_before, metric="euclidean")
    dist_after = pdist(X_after, metric="euclidean")

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
    distances = pdist(centroids, metric="euclidean")

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
            stat, _ = levene(*groups, center="median")
            if not np.isnan(stat):
                levene_stats.append(stat)
        except Exception:
            continue

    if len(levene_stats) == 0:
        return 0.0

    return np.median(levene_stats)


class ComBatMetricsMixin:
    """Mixin providing batch effect metrics for the ComBat wrapper."""

    @property
    def metrics_(self) -> dict[str, Any] | None:
        """Return cached metrics from last fit_transform with compute_metrics=True.

        Returns
        -------
        dict or None
            Cached metrics dictionary, or None if no metrics have been computed.
        """
        return getattr(self, "_metrics_cache", None)

    def compute_batch_metrics(
        self,
        X: ArrayLike,
        batch: ArrayLike | None = None,
        *,
        pca_components: int | None = None,
        k_neighbors: list[int] | None = None,
        kbet_k0: int | None = None,
        lisi_perplexity: int = 30,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
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
        dict
            Dictionary with three main keys:

            - ``batch_effect``: Silhouette, Davies-Bouldin, kBET, LISI, variance ratio
              (each with 'before' and 'after' values)
            - ``preservation``: k-NN preservation fractions, distance correlation
            - ``alignment``: Centroid distance, Levene statistic (each with
              'before' and 'after' values)

        Raises
        ------
        ValueError
            If the model is not fitted or if pca_components is invalid.
        """
        if not hasattr(self._model, "_gamma_star"):
            raise ValueError(
                "This ComBat instance is not fitted yet. Call 'fit' before 'compute_batch_metrics'."
            )

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        idx = X.index

        if batch is None:
            batch_vec = self._subset(self.batch, idx)
        else:
            if isinstance(batch, (pd.Series, pd.DataFrame)):
                batch_vec = batch.loc[idx] if hasattr(batch, "loc") else batch
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
        if k_neighbors is None:
            k_neighbors = [5, 10, 50]

        # Validate and apply PCA if requested
        if pca_components is not None:
            max_components = min(n_samples, n_features)
            if pca_components >= max_components:
                raise ValueError(
                    f"pca_components={pca_components} must be less than "
                    f"min(n_samples, n_features)={max_components}."
                )
            X_before_pca, X_after_pca, _ = _compute_pca_embedding(X_before, X_after, pca_components)
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
            "batch_effect": {
                "silhouette": {
                    "before": silhouette_before,
                    "after": silhouette_after,
                },
                "davies_bouldin": {
                    "before": db_before,
                    "after": db_after,
                },
                "kbet": {
                    "before": kbet_before,
                    "after": kbet_after,
                },
                "lisi": {
                    "before": lisi_before,
                    "after": lisi_after,
                    "max_value": n_batches,
                },
                "variance_ratio": {
                    "before": var_ratio_before,
                    "after": var_ratio_after,
                },
            },
            "preservation": {
                "knn": knn_results,
                "distance_correlation": dist_corr,
            },
            "alignment": {
                "centroid_distance": {
                    "before": centroid_before,
                    "after": centroid_after,
                },
                "levene_statistic": {
                    "before": levene_before,
                    "after": levene_after,
                },
            },
        }

        return metrics

    def feature_batch_importance(
        self,
        mode: Literal["magnitude", "distribution"] = "magnitude",
    ) -> pd.DataFrame:
        """Compute per-feature batch effect magnitude.

        Returns DataFrame with columns: 'location', 'scale', 'combined'
        - location: RMS of gamma across batches (standardized mean shifts)
        - scale: RMS of log(delta) across batches (log-fold variance change)
        - combined: sqrt(location**2 + scale**2) - Euclidean norm treating
          location and scale as orthogonal dimensions

        Using RMS (root mean square) provides L2-consistent aggregation.
        Using log(delta) ensures symmetry: delta=2 and delta=0.5
        represent equally strong effects in opposite directions.

        Parameters
        ----------
        mode : {'magnitude', 'distribution'}, default='magnitude'
            - 'magnitude': Returns L2-consistent absolute batch effect magnitudes.
              Suitable for ranking, thresholding, and cross-dataset comparison.
            - 'distribution': Returns column-wise normalized proportions (each column
              sums to 1, values in range [0, 1]), representing the relative contribution
              of each feature to the total location, scale, or combined batch effect.
              Note: normalization is applied independently to each column, so the
              Euclidean relationship (combined**2 = location**2 + scale**2) no longer holds.

        Returns
        -------
        pd.DataFrame
            DataFrame with index=feature names, columns=['location', 'scale', 'combined'],
            sorted by 'combined' descending.

        Raises
        ------
        ValueError
            If the model is not fitted or if mode is invalid.
        """
        if not hasattr(self._model, "_gamma_star"):
            raise ValueError(
                "This ComBat instance is not fitted yet. "
                "Call 'fit' before 'feature_batch_importance'."
            )

        if mode not in ["magnitude", "distribution"]:
            raise ValueError(f"mode must be 'magnitude' or 'distribution', got '{mode}'")

        feature_names = self._model._grand_mean.index
        gamma_star = self._model._gamma_star
        delta_star = self._model._delta_star

        # Location effect: RMS of gamma across batches (L2 aggregation)
        location = np.sqrt((gamma_star**2).mean(axis=0))

        # Scale effect: RMS of log(delta) across batches
        if not self.mean_only:
            scale = np.sqrt((np.log(delta_star) ** 2).mean(axis=0))
        else:
            scale = np.zeros_like(location)

        # Euclidean to treat location and scale as orthogonal dimensions
        combined = np.sqrt(location**2 + scale**2)

        if mode == "distribution":
            # Normalize each column independently to sum to 1
            location_sum = location.sum()
            scale_sum = scale.sum()
            combined_sum = combined.sum()

            location = location / location_sum if location_sum > 0 else location
            scale = scale / scale_sum if scale_sum > 0 else scale
            combined = combined / combined_sum if combined_sum > 0 else combined

        return pd.DataFrame(
            {
                "location": location,
                "scale": scale,
                "combined": combined,
            },
            index=feature_names,
        ).sort_values("combined", ascending=False)

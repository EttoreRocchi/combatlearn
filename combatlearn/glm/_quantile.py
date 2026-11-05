"""Per-observation quantile maps from a batch-specific to a batch-free distribution.

Strictly per cell: nothing is pooled across the samples being transformed, so a
held-out sample maps identically alone or alongside others.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import nbinom

from ..core import FloatArray

# Lower-tail probability this close to 1 would map to an infinite quantile
_TAIL_TOL = 1e-4


def map_to_batch_free_nb(
    counts: FloatArray,
    old_mu: FloatArray,
    old_phi: FloatArray,
    new_mu: FloatArray,
    new_phi: FloatArray,
) -> FloatArray:
    """Map counts from their batch-specific NB distribution to the batch-free one.

    Parameters
    ----------
    counts : ndarray of shape (n_features, n_samples)
        Observed integer counts.
    old_mu, new_mu : ndarray of shape (n_features, n_samples)
        Fitted and batch-free means.
    old_phi, new_phi : ndarray of shape (n_features,)
        Batch-specific and batch-free dispersions.

    Returns
    -------
    ndarray of shape (n_features, n_samples)
        Adjusted counts. Counts of 0 and 1, and cells in the extreme upper
        tail, are left unchanged.
    """
    adjusted = np.array(counts, dtype=np.float64, copy=True)
    mapped = counts > 1
    if not mapped.any():
        return adjusted

    old_size = np.broadcast_to(np.asarray(old_phi, dtype=np.float64)[:, None], counts.shape)
    new_size = np.broadcast_to(np.asarray(new_phi, dtype=np.float64)[:, None], counts.shape)
    old_size = 1.0 / old_size[mapped]
    new_size = 1.0 / new_size[mapped]

    old_p = old_size / (old_size + old_mu[mapped])
    new_p = new_size / (new_size + new_mu[mapped])

    lower_tail = nbinom.cdf(counts[mapped] - 1, old_size, old_p)
    adjusted[mapped] = np.where(
        np.abs(lower_tail - 1.0) < _TAIL_TOL,
        counts[mapped],
        1.0 + nbinom.ppf(lower_tail, new_size, new_p),
    )
    return adjusted


def map_to_batch_free_beta(
    values: FloatArray,
    old_mu: FloatArray,
    old_phi: FloatArray,
    new_mu: FloatArray,
    new_phi: FloatArray,
) -> FloatArray:
    """Map beta-values from their batch-specific beta distribution to the batch-free one.

    Parameters
    ----------
    values : ndarray of shape (n_features, n_samples)
        Observed proportions, strictly inside ``(0, 1)``.
    old_mu, new_mu : ndarray of shape (n_features, n_samples)
        Fitted and batch-free means.
    old_phi, new_phi : ndarray of shape (n_features, n_samples)
        Batch-specific and batch-free precisions.

    Returns
    -------
    ndarray of shape (n_features, n_samples)
        Adjusted proportions, again strictly inside ``(0, 1)``. Cells whose
        parameters are not usable are passed through unchanged.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        lower_tail = beta_dist.cdf(values, old_mu * old_phi, (1.0 - old_mu) * old_phi)
        target = beta_dist.ppf(lower_tail, new_mu * new_phi, (1.0 - new_mu) * new_phi)
    usable = np.isfinite(lower_tail) & np.isfinite(target)
    return np.asarray(np.where(usable, target, values), dtype=np.float64)

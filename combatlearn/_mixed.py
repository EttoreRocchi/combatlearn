"""Random-intercept linear mixed model fitting for Longitudinal ComBat.

Fits, for every feature (column of ``Y``) simultaneously, the model

    y = X @ beta + Z @ u + eps,    u ~ N(0, tau^2 I),   eps ~ N(0, sigma^2 I)

where ``Z`` is the subject (random-intercept) indicator matrix. Estimation is
by REML, profiling out ``beta`` and ``sigma^2`` so that only the scalar
variance ratio ``lambda = tau^2 / sigma^2`` is optimized per feature.

The single random-intercept grouping makes the marginal covariance
``V = sigma^2 (I + lambda Z Z^T)`` block diagonal (one block per subject), so
``V^{-1}`` is available in closed form via Sherman-Morrison and every operation
is O(n) per feature. The variance ratio is found by a vectorized grid search
over ``lambda`` followed by a parabolic refinement, after which the fixed
effects, residual variance, and random-intercept BLUPs are computed in a single
per-feature pass.

This keeps Longitudinal ComBat dependency-free (numpy only); correctness is
cross-checked against ``statsmodels.MixedLM`` in the test suite.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def _group_sums(M: FloatArray, group_idx: IntArray, n_groups: int) -> FloatArray:
    """Sum the rows of ``M`` (n, c) within each group, returning (n_groups, c)."""
    out = np.zeros((n_groups, M.shape[1]), dtype=np.float64)
    np.add.at(out, group_idx, M)
    return out


def _w_apply(M: FloatArray, factor: FloatArray, group_idx: IntArray, n_groups: int) -> FloatArray:
    """Apply ``W = (I + lambda Z Z^T)^{-1}`` to the columns of ``M``.

    ``factor[k] = lambda / (1 + lambda * n_k)`` is the Sherman-Morrison
    per-group correction (``n_k`` = number of observations in group ``k``).
    """
    gs = _group_sums(M, group_idx, n_groups)
    return M - factor[group_idx][:, None] * gs[group_idx]


def _reml_objective(
    lam: float,
    X: FloatArray,
    Y: FloatArray,
    group_idx: IntArray,
    n_groups: int,
    n_k: FloatArray,
    n: int,
    p: int,
    ridge: float,
) -> FloatArray:
    """Profiled REML objective (to minimize) evaluated for all features at ``lam``."""
    factor = lam / (1.0 + lam * n_k)
    WX = _w_apply(X, factor, group_idx, n_groups)
    A = X.T @ WX
    A.flat[:: p + 1] += ridge
    beta = np.linalg.solve(A, X.T @ _w_apply(Y, factor, group_idx, n_groups))
    R = Y - X @ beta
    WR = _w_apply(R, factor, group_idx, n_groups)
    rss = np.maximum(np.einsum("ng,ng->g", R, WR), 1e-12)
    logdet_v = float(np.sum(np.log1p(lam * n_k)))
    _, logdet_a = np.linalg.slogdet(A)
    result = logdet_v + (n - p) * np.log(rss) + float(logdet_a)
    return np.asarray(result, dtype=np.float64)


def fit_random_intercept(
    X: FloatArray,
    Y: FloatArray,
    group_idx: IntArray,
    n_groups: int,
    n_k: FloatArray,
    *,
    eps: float = 1e-8,
    n_grid: int = 80,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Fit a random-intercept LME per feature by REML.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Fixed-effects design matrix (shared across features).
    Y : ndarray of shape (n, n_features)
        Response matrix; one column per feature.
    group_idx : ndarray of shape (n,)
        Integer group (subject) code in ``[0, n_groups)`` for each observation.
    n_groups : int
        Number of distinct subjects.
    n_k : ndarray of shape (n_groups,)
        Number of observations per subject.
    eps : float, default=1e-8
        Numerical jitter; also scales the diagonal ridge on ``X^T W X``.
    n_grid : int, default=80
        Number of log-spaced ``lambda`` grid points for the variance-ratio search.

    Returns
    -------
    beta : ndarray of shape (p, n_features)
        Estimated fixed-effects coefficients.
    blup : ndarray of shape (n_groups, n_features)
        Best linear unbiased predictors of the subject random intercepts.
    sigma2 : ndarray of shape (n_features,)
        Estimated residual variance per feature.
    lam_opt : ndarray of shape (n_features,)
        Estimated variance ratio ``tau^2 / sigma^2`` per feature.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    group_idx = np.asarray(group_idx, dtype=np.intp)
    n_k = np.asarray(n_k, dtype=np.float64)
    n, p = X.shape
    n_features = Y.shape[1]
    ridge = eps * float(np.trace(X.T @ X)) / max(p, 1)

    # Vectorized grid search for the variance ratio (per feature).
    log_grid = np.linspace(-4.0, 4.0, n_grid)
    lam_grid = np.concatenate([[0.0], 10.0**log_grid])
    fvals = np.empty((lam_grid.size, n_features), dtype=np.float64)
    for i, lam in enumerate(lam_grid):
        fvals[i] = _reml_objective(lam, X, Y, group_idx, n_groups, n_k, n, p, ridge)
    best = np.argmin(fvals, axis=0)
    lam_opt = lam_grid[best].astype(np.float64)

    # Vectorized parabolic refinement (log10 space) for interior positive optima.
    interior = (best >= 2) & (best <= n_grid - 1)
    if np.any(interior):
        bi = best[interior]
        f_lo = fvals[bi - 1, interior]
        f0 = fvals[bi, interior]
        f_hi = fvals[bi + 1, interior]
        denom = f_lo - 2.0 * f0 + f_hi
        delta = np.where(np.abs(denom) > 1e-12, 0.5 * (f_lo - f_hi) / denom, 0.0)
        delta = np.clip(delta, -1.0, 1.0)
        step = log_grid[1] - log_grid[0]
        lam_opt[interior] = 10.0 ** (log_grid[bi - 1] + delta * step)

    # Final per-feature fixed effects, residual variance, and random-intercept BLUPs.
    beta = np.empty((p, n_features), dtype=np.float64)
    sigma2 = np.empty(n_features, dtype=np.float64)
    blup = np.zeros((n_groups, n_features), dtype=np.float64)
    denom_sigma = max(n - p, 1)
    for g in range(n_features):
        lam = float(lam_opt[g])
        factor = lam / (1.0 + lam * n_k)
        A = X.T @ _w_apply(X, factor, group_idx, n_groups)
        A.flat[:: p + 1] += ridge
        y = Y[:, g : g + 1]
        b = np.linalg.solve(A, X.T @ _w_apply(y, factor, group_idx, n_groups))[:, 0]
        beta[:, g] = b
        r = Y[:, g] - X @ b
        Wr = _w_apply(r[:, None], factor, group_idx, n_groups)[:, 0]
        sigma2[g] = float(r @ Wr) / denom_sigma
        s_k = np.zeros(n_groups, dtype=np.float64)
        np.add.at(s_k, group_idx, r)
        blup[:, g] = lam * s_k / (1.0 + lam * n_k)

    return beta, blup, sigma2, lam_opt

"""Negative-binomial GLM machinery for ComBat-seq.

NB log-linear model with genewise dispersions (McCarthy, Chen & Smyth 2012) and the
Cox-Reid (1987) adjusted profile likelihood, reimplemented on numpy/scipy; edgeR,
sva and InMoose are test oracles only. Features in rows, samples in columns.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.linalg import qr, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from ..core import FloatArray
from ._interpolate import maximize_interpolant

_LOW = 1e-10
_LOG_LOW = math.log(_LOW)
_MILDLY_LOW = 1e-8
_SUPREMELY_LOW = 1e-13
_RIDICULOUSLY_LOW = 1e-100


def _observation_deviance(y: FloatArray, mu: FloatArray, phi: FloatArray) -> FloatArray:
    """Per-observation negative-binomial deviance.

    Poisson and Gamma limits for tiny ``phi`` and huge ``phi * mu`` avoid
    cancelling two large logarithms in the plain NB form.
    """
    y = y + _MILDLY_LOW
    mu = mu + _MILDLY_LOW
    resid = y - mu
    product = mu * phi
    log_ratio = np.log(y / mu)

    with np.errstate(divide="ignore", invalid="ignore"):
        poisson = 2.0 * (
            y * log_ratio - resid - 0.5 * resid**2 * phi * (1.0 + phi * (2.0 / 3.0 * resid - y))
        )
        gamma = 2.0 * (resid / mu - log_ratio) * mu / (1.0 + product)
        inv_phi = 1.0 / np.where(phi > 0.0, phi, 1.0)
        nbinom = 2.0 * (y * log_ratio + (y + inv_phi) * np.log((mu + inv_phi) / (y + inv_phi)))

    return np.asarray(
        np.where(phi < 1e-4, poisson, np.where(product > 1e6, gamma, nbinom)), dtype=np.float64
    )


def nb_deviance(y: FloatArray, mu: FloatArray, phi: FloatArray) -> FloatArray:
    """Genewise negative-binomial deviance (sum over samples)."""
    return np.asarray(_observation_deviance(y, mu, phi).sum(axis=1), dtype=np.float64)


def _design_cells(design: FloatArray) -> tuple[npt.NDArray[np.intp], int]:
    """Map each design row to a group index, collapsing identical rows.

    Rows are hashed by a weighted mean with irrational weights, avoiding a
    pairwise comparison of rows.
    """
    weights = ((np.e + np.pi) / 5.0) ** np.arange(design.shape[1])
    keys = (design * weights).mean(axis=1)
    uniq = np.unique(keys)
    return np.asarray(np.searchsorted(uniq, keys), dtype=np.intp), len(uniq)


def _one_group(
    y: FloatArray,
    offset: FloatArray,
    dispersion: FloatArray,
    coef_start: FloatArray | None = None,
    maxit: int = 50,
    tol: float = 1e-10,
) -> FloatArray:
    """Fit an intercept-only NB GLM to every row by Fisher scoring."""
    n_cols = y.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if coef_start is None:
            total = np.where(y > _LOW, y / np.exp(offset), 0.0).sum(axis=1)
            beta = np.log(total / n_cols)
        else:
            beta = np.array(coef_start, dtype=np.float64, copy=True)

    nonzero = (y > _LOW).any(axis=1)
    beta = np.where(nonzero, beta, -np.inf)

    poisson = (dispersion == 0.0).all(axis=1) & nonzero
    if poisson.any():
        with np.errstate(divide="ignore"):
            beta[poisson] = np.log(y[poisson].sum(axis=1) / np.exp(offset[poisson]).sum(axis=1))

    active = nonzero & ~poisson
    for _ in range(maxit):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        mu = np.exp(beta[idx, None] + offset[idx])
        denom = 1.0 + mu * dispersion[idx]
        step = ((y[idx] - mu) / denom).sum(axis=1) / (mu / denom).sum(axis=1)
        beta[idx] += step
        active[idx[np.abs(step) < tol]] = False
    return beta


def fit_group_means(
    y: FloatArray,
    design: FloatArray,
    dispersion: FloatArray,
    offset: FloatArray,
    coef_start: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Fit an NB GLM to every row as one mean per design cell.

    Exact when the design has as many distinct rows as columns: each cell is
    fitted on its own and the cell means are solved back onto the design.
    """
    group, n_groups = _design_cells(design)
    first = np.array([int(np.flatnonzero(group == g)[0]) for g in range(n_groups)])
    design_unique = design[first]

    indicator = (
        int((design_unique == 1).sum()) == n_groups
        and int((design_unique == 0).sum()) == (n_groups - 1) * n_groups
    )
    if not indicator and coef_start is not None:
        coef_start = coef_start @ design_unique.T

    beta: FloatArray = np.zeros((y.shape[0], n_groups))
    for g in range(n_groups):
        cols = np.flatnonzero(group == g)
        beta[:, g] = _one_group(
            y[:, cols],
            offset[:, cols],
            dispersion[:, cols],
            coef_start=None if coef_start is None else coef_start[:, g],
        )

    beta = np.where(beta > -1e8, beta, -1e8)
    mu = np.exp(offset + beta[:, group])
    if not indicator:
        beta = np.linalg.solve(design_unique, beta.T).T
    return beta, mu


def _batch_cholesky(a: FloatArray) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
    """Cholesky factor, pivots and success flag for a stack of small matrices.

    The pivots are the ``D`` of ``L D L'``, so their product is the determinant;
    the flag is False where a pivot was not positive.
    """
    n_mat, p, _ = a.shape
    ell: FloatArray = np.zeros_like(a)
    piv: FloatArray = np.zeros((n_mat, p))
    ok = np.ones(n_mat, dtype=bool)
    for i in range(p):
        s = a[:, i, i] - np.einsum("mk,mk->m", ell[:, i, :i], ell[:, i, :i])
        bad = ~(s > 0.0)
        ok &= ~bad
        piv[:, i] = s
        ell[:, i, i] = np.sqrt(np.where(bad, 1.0, s))
        if i + 1 < p:
            cross = np.einsum("mjk,mk->mj", ell[:, i + 1 :, :i], ell[:, i, :i])
            ell[:, i + 1 :, i] = (a[:, i + 1 :, i] - cross) / ell[:, i, i, None]
    return ell, piv, ok


def _chol_solve(ell: FloatArray, b: FloatArray) -> FloatArray:
    """Solve ``L L' x = b`` for a stack of lower-triangular factors."""
    p = ell.shape[1]
    z = np.zeros_like(b)
    for i in range(p):
        z[:, i] = (b[:, i] - np.einsum("mk,mk->m", ell[:, i, :i], z[:, :i])) / ell[:, i, i]
    x = np.zeros_like(b)
    for i in range(p - 1, -1, -1):
        x[:, i] = (z[:, i] - np.einsum("mk,mk->m", ell[:, i + 1 :, i], x[:, i + 1 :])) / ell[
            :, i, i
        ]
    return x


def _null_model_start(
    y: FloatArray, offset: FloatArray, dispersion: FloatArray, design: FloatArray
) -> FloatArray:
    """Null-model starting coefficients: the fitted intercept projected onto ``design``."""
    q, r = qr(design, mode="economic")
    lib = np.exp(offset)
    weight = lib / (1.0 + dispersion * lib)
    with np.errstate(divide="ignore", invalid="ignore"):
        level = np.log((y * weight / lib).sum(axis=1) / weight.sum(axis=1))
    basis = solve_triangular(r, q.T @ np.ones(design.shape[0]), check_finite=False)
    return np.asarray(level[:, None] * basis[None, :], dtype=np.float64)


def fit_damped_irls(
    y: FloatArray,
    design: FloatArray,
    dispersion: FloatArray,
    offset: FloatArray,
    coef_start: FloatArray | None = None,
    maxit: int = 250,
    tol: float = 1e-6,
) -> tuple[FloatArray, FloatArray]:
    """Fit a general NB GLM to every row by Levenberg-damped Fisher scoring.

    Damping grows until the deviance stops rising, which keeps the fit stable
    when the information matrix is near-singular.
    """
    n_rows = y.shape[0]
    n_coef = design.shape[1]
    beta = (
        _null_model_start(y, offset, dispersion, design)
        if coef_start is None
        else np.array(coef_start, dtype=np.float64, copy=True)
    )

    mu = np.exp(beta @ design.T + offset)
    dev = nb_deviance(y, mu, dispersion)

    y_max = y.max(axis=1)
    empty = y_max < _LOW
    beta[empty, :] = np.nan
    mu[empty, :] = 0.0
    dev[empty] = 0.0

    remaining = ~empty
    lam: FloatArray = np.zeros(n_rows, dtype=np.float64)
    converged = np.zeros(n_rows, dtype=bool)
    dbeta: FloatArray = np.zeros((n_rows, n_coef))

    for it in range(maxit):
        if not remaining.any():
            break

        denom = 1.0 + mu * dispersion
        working = mu / denom
        score = ((y - mu) / denom) @ design
        info = np.einsum("ji,mj,jk->mik", design, working, design)
        max_info = np.diagonal(info, axis1=1, axis2=2).max(axis=1)
        if it == 0:
            lam = np.asarray(np.maximum(max_info * 1e-6, _SUPREMELY_LOW), dtype=np.float64)

        lev_step = np.zeros(n_rows, dtype=int)
        low_dev = np.zeros(n_rows, dtype=bool)
        pending = remaining.copy()

        while pending.any():
            lev_step[pending] += 1

            eye = np.eye(n_coef)
            while True:
                ell, _, ok = _batch_cholesky(info + lam[:, None, None] * eye)
                broken = pending & ~ok
                if not broken.any():
                    break
                lam[broken] *= 10.0
                lam[broken & (lam <= 0.0)] = _RIDICULOUSLY_LOW
            dbeta = _chol_solve(ell, score)

            beta_new = beta + dbeta
            mu_new = np.exp(beta_new @ design.T + offset)
            dev_new = nb_deviance(y, mu_new, dispersion)

            with np.errstate(invalid="ignore", divide="ignore"):
                # Masked by `pending`: rows that already accepted a step must not be flagged.
                low_dev |= pending & ((dev_new / y_max) < _SUPREMELY_LOW)
            accept = pending & ((dev_new <= dev) | low_dev)
            beta[accept, :] = beta_new[accept, :]
            mu[accept, :] = mu_new[accept, :]
            dev[accept] = dev_new[accept]
            pending &= ~accept

            lam[pending] *= 2.0
            lam[pending & (lam <= 0.0)] = _RIDICULOUSLY_LOW
            with np.errstate(divide="ignore", invalid="ignore"):
                converged[pending & ((lam / max_info) > (1.0 / _SUPREMELY_LOW))] = True
            pending &= ~converged

        remaining &= ~converged
        remaining &= ~low_dev
        remaining &= ~((score * dbeta).sum(axis=1) < tol)
        lam[remaining & (lev_step == 1)] /= 10.0

    return beta, mu


def _augment_with_prior(
    y: FloatArray, offset: FloatArray, prior_count: float
) -> tuple[FloatArray, FloatArray]:
    """Add a library-size-proportional prior count, and adjust the offsets to match."""
    lib = np.exp(offset)
    prior = prior_count * lib / lib.mean(axis=1)[:, None]
    return y + prior, np.log(lib + 2.0 * prior)


class NBFit(NamedTuple):
    """Result of :func:`nb_glm_fit`.

    ``coefficients`` are shrunk when ``prior_count > 0`` (else equal to
    ``unshrunk_coefficients``); ``fitted_values`` always come from the unshrunk fit.
    """

    coefficients: FloatArray
    fitted_values: FloatArray
    unshrunk_coefficients: FloatArray


def nb_glm_fit(
    y: FloatArray,
    design: FloatArray,
    dispersion: FloatArray,
    offset: FloatArray,
    prior_count: float = 0.0,
    coef_start: FloatArray | None = None,
) -> NBFit:
    """Fit a negative-binomial log-linear model to every row of ``y``.

    Parameters
    ----------
    y : ndarray of shape (n_features, n_samples)
        Counts.
    design : ndarray of shape (n_samples, n_coefficients)
        Full-column-rank design matrix.
    dispersion : ndarray of shape (n_features, n_samples)
        Known dispersion for every observation.
    offset : ndarray of shape (n_features, n_samples)
        Log-scale offset, normally the log library size.
    prior_count : float, default=0.0
        When positive, the returned *coefficients* come from a refit on
        prior-count-augmented data, which shrinks log-fold-changes toward zero.
        The fitted values always come from the unaugmented fit.
    coef_start : ndarray, optional
        Starting coefficients.

    Returns
    -------
    NBFit
        Shrunk coefficients, fitted values and unshrunk coefficients.
    """
    _, n_groups = _design_cells(design)
    if n_groups == design.shape[1]:
        coef, mu = fit_group_means(y, design, dispersion, offset, coef_start=coef_start)
    else:
        coef, mu = fit_damped_irls(y, design, dispersion, offset, coef_start=coef_start)

    shrunk = coef
    if prior_count > 0.0:
        y_prior, offset_prior = _augment_with_prior(y, offset, prior_count)
        shrunk = nb_glm_fit(y_prior, design, dispersion, offset_prior, prior_count=0.0).coefficients
    return NBFit(shrunk, mu, coef)


def adjusted_profile_likelihood(
    dispersion: FloatArray | float,
    y: FloatArray,
    design: FloatArray,
    offset: FloatArray,
    coef_start: FloatArray | None = None,
    return_coef: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Cox-Reid adjusted profile log-likelihood of the dispersion, per feature.

    The profile likelihood at the conditional MLE of the coefficients, penalised
    by half the log-determinant of the information (Cox & Reid 1987; McCarthy,
    Chen & Smyth 2012).
    """
    disp = np.broadcast_to(np.asarray(dispersion, dtype=np.float64).reshape(-1, 1), y.shape)
    coef, mu, _ = nb_glm_fit(y, design, disp, offset, prior_count=0.0, coef_start=coef_start)

    valid = mu > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        size = 1.0 / np.where(disp > 0.0, disp, 1.0)
        log_mu_size = np.log(mu + size)
        loglik = (
            y * np.log(mu)
            - y * log_mu_size
            + size * np.log(size)
            - size * log_mu_size
            + gammaln(y + size)
            - gammaln(y + 1.0)
            - gammaln(size)
        )
        poisson_loglik = y * np.log(mu) - mu - gammaln(y + 1.0)
    loglik = np.where(disp > 0.0, loglik, poisson_loglik)
    apl = np.where(valid, loglik, 0.0).sum(axis=1)

    working = np.where(valid, mu / (1.0 + disp * mu), 0.0)
    if design.shape[1] == 1:
        with np.errstate(divide="ignore"):
            apl = apl - 0.5 * np.log(np.abs(working.sum(axis=1)))
    else:
        info = np.einsum("ji,mj,jk->mik", design, working, design)
        _, piv, _ = _batch_cholesky(info)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_piv = np.where((piv < _LOW) | ~np.isfinite(piv), _LOG_LOW, np.log(piv))
        apl = apl - 0.5 * log_piv.sum(axis=1)

    apl = np.asarray(apl, dtype=np.float64)
    return (apl, coef) if return_coef else apl


def common_dispersion(
    y: FloatArray,
    design: FloatArray,
    offset: FloatArray,
    interval: tuple[float, float] = (0.0, 4.0),
    tol: float = 1e-5,
    min_row_sum: int = 5,
) -> float:
    """Common NB dispersion maximizing the summed Cox-Reid adjusted profile likelihood.

    Features below ``min_row_sum`` total counts are excluded: they bias it downward.
    """
    keep = y.sum(axis=1) >= min_row_sum
    if not keep.any():
        raise ValueError(
            f"no feature reaches the minimum total count of {min_row_sum} needed to "
            f"estimate a dispersion. Check that X holds raw counts, not normalized or "
            f"log-transformed values."
        )
    y, offset = y[keep], offset[keep]

    def objective(par: float) -> float:
        apl = adjusted_profile_likelihood(par**4, y, design, offset)
        return -float(np.sum(np.asarray(apl)))

    result = minimize_scalar(
        objective,
        bounds=(interval[0] ** 0.25, interval[1] ** 0.25),
        # Pinned: SciPy < 1.10 ignores `bounds` unless method='bounded'.
        method="bounded",
        options={"xatol": tol},
    )
    return float(result.x) ** 4


def featurewise_dispersion(
    y: FloatArray,
    design: FloatArray,
    dispersion: float,
    offset: FloatArray,
    min_row_sum: int = 5,
    grid_npts: int = 11,
    grid_range: tuple[float, float] = (-6.0, 6.0),
) -> FloatArray:
    """Per-feature NB dispersion by Cox-Reid APL on a log2 grid around ``dispersion``.

    Features below ``min_row_sum`` total counts keep the common value.
    """
    n_features = y.shape[0]
    disp = np.full(n_features, float(dispersion))

    informative = y.sum(axis=1) >= min_row_sum
    if not informative.all():
        if informative.any():
            disp[informative] = featurewise_dispersion(
                y[informative],
                design,
                dispersion,
                offset[informative],
                min_row_sum=0,
                grid_npts=grid_npts,
                grid_range=grid_range,
            )
        return disp

    spline_pts = np.array(
        [
            grid_range[0] + i * (grid_range[1] - grid_range[0]) / (grid_npts - 1)
            for i in range(grid_npts)
        ]
    )
    apl = np.zeros((n_features, grid_npts))
    for i, pt in enumerate(spline_pts):
        apl[:, i] = np.asarray(adjusted_profile_likelihood(disp * 2.0**pt, y, design, offset))

    return np.asarray(disp * 2.0 ** maximize_interpolant(spline_pts, apl), dtype=np.float64)

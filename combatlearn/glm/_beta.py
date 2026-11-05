"""Beta-regression machinery for ComBat-met.

Logit-link beta regression at known precision (Ferrari & Cribari-Neto), fitted by
damped Fisher scoring, with the precision from a Cox-Reid adjusted profile
likelihood. Features in rows, samples in columns.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import digamma, gammaln, polygamma

from ..core import FloatArray
from ._interpolate import maximize_interpolant
from ._nb import _batch_cholesky, _chol_solve

_LOW = 1e-10
_LOG_LOW = math.log(_LOW)
_SUPREMELY_LOW = 1e-13
_RIDICULOUSLY_LOW = 1e-100


def _trigamma(x: FloatArray) -> FloatArray:
    return np.asarray(polygamma(1, x), dtype=np.float64)


def _beta_neg_loglik(y: FloatArray, mu: FloatArray, phi: FloatArray) -> FloatArray:
    """Per-observation negative log-likelihood of the beta density.

    Degenerate fits get a large finite penalty so damped steps stay comparable.
    """
    a = mu * phi
    b = (1.0 - mu) * phi
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = (
            gammaln(phi)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y)
            + (b - 1.0) * np.log1p(-y)
        )
    degenerate = (mu <= 0.0) | (mu >= 1.0) | (phi <= 0.0)
    return np.asarray(np.where(degenerate, 1e30, -loglik), dtype=np.float64)


class BetaFit(NamedTuple):
    """Result of :func:`beta_glm_fit`: logit-scale coefficients and fitted means."""

    coefficients: FloatArray
    fitted_values: FloatArray


def _logit(x: FloatArray) -> FloatArray:
    return np.asarray(np.log(x) - np.log1p(-x), dtype=np.float64)


def _expit(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def beta_glm_fit(
    y: FloatArray,
    design: FloatArray,
    phi: FloatArray,
    coef_start: FloatArray | None = None,
    maxit: int = 200,
    tol: float = 1e-6,
) -> BetaFit:
    """Fit a logit-link beta regression to every row of ``y`` at known precision.

    Parameters
    ----------
    y : ndarray of shape (n_features, n_samples)
        Proportions strictly inside ``(0, 1)``.
    design : ndarray of shape (n_samples, n_coefficients)
        Full-column-rank design matrix.
    phi : ndarray of shape (n_features, n_samples)
        Known precision for every observation.
    coef_start : ndarray, optional
        Starting coefficients. Defaults to the least-squares fit of
        ``logit(y)`` on ``design``.
    maxit, tol : int, float
        Iteration cap and convergence tolerance on the score/step inner product.
    """
    n_rows = y.shape[0]
    n_coef = design.shape[1]

    if coef_start is None:
        beta = np.asarray(np.linalg.lstsq(design, _logit(y).T, rcond=None)[0].T, dtype=np.float64)
    else:
        beta = np.array(coef_start, dtype=np.float64, copy=True)

    mu = _expit(beta @ design.T)
    dev = _beta_neg_loglik(y, mu, phi).sum(axis=1)

    degenerate = ~((y > _LOW) & (y < 1.0 - _LOW)).any(axis=1)
    beta[degenerate, :] = np.nan
    mu[degenerate, :] = 0.5
    dev[degenerate] = 0.0

    remaining = ~degenerate
    max_info: FloatArray = np.full(n_rows, -1.0)
    lam: FloatArray = np.zeros(n_rows)
    logit_y = _logit(np.clip(y, _LOW, 1.0 - _LOW))

    for it in range(maxit):
        if not remaining.any():
            break

        a = mu * phi
        b = (1.0 - mu) * phi
        dmu = mu * (1.0 - mu)
        working = phi**2 * (_trigamma(a) + _trigamma(b)) * dmu**2
        score = ((logit_y - digamma(a) + digamma(b)) * phi * dmu) @ design

        info = np.einsum("ji,mj,jk->mik", design, working, design)
        max_info = np.maximum(max_info, np.diagonal(info, axis1=1, axis2=2).max(axis=1))
        if it == 0:
            lam = np.asarray(np.maximum(max_info * 1e-6, _SUPREMELY_LOW), dtype=np.float64)

        lev_step = np.zeros(n_rows, dtype=int)
        failed = np.zeros(n_rows, dtype=bool)
        low_dev = np.zeros(n_rows, dtype=bool)
        dbeta: FloatArray = np.zeros((n_rows, n_coef))
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
            mu_new = _expit(beta_new @ design.T)
            dev_new = _beta_neg_loglik(y, mu_new, phi).sum(axis=1)

            low_dev |= pending & (dev_new < _SUPREMELY_LOW)
            accept = pending & ((dev_new <= dev) | low_dev)
            beta[accept, :] = beta_new[accept, :]
            mu[accept, :] = mu_new[accept, :]
            dev[accept] = dev_new[accept]
            pending &= ~accept

            lam[pending] *= 2.0
            lam[pending & (lam <= 0.0)] = _RIDICULOUSLY_LOW
            with np.errstate(divide="ignore", invalid="ignore"):
                failed |= pending & ((lam / max_info) > (1.0 / _SUPREMELY_LOW))
            pending &= ~failed

        remaining &= ~failed
        remaining &= ~low_dev
        remaining &= ~((score * dbeta).sum(axis=1) < tol)
        lam[remaining & (lev_step == 1)] /= 10.0

    return BetaFit(beta, mu)


def adjusted_profile_likelihood_beta(
    phi: FloatArray | float,
    y: FloatArray,
    design: FloatArray,
    coef_start: FloatArray | None = None,
    maxit: int = 200,
    tol: float = 1e-8,
    return_coef: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    """Cox-Reid adjusted profile log-likelihood of the beta precision, per feature."""
    phi_mat = np.broadcast_to(np.asarray(phi, dtype=np.float64).reshape(-1, 1), y.shape)
    coef, mu = beta_glm_fit(y, design, phi_mat, coef_start=coef_start, maxit=maxit, tol=tol)

    usable = (mu > 0.0) & (mu < 1.0)
    a = mu * phi_mat
    b = (1.0 - mu) * phi_mat
    with np.errstate(divide="ignore", invalid="ignore"):
        loglik = (
            gammaln(phi_mat)
            - gammaln(a)
            - gammaln(b)
            + (a - 1.0) * np.log(y)
            + (b - 1.0) * np.log1p(-y)
        )
    apl = np.where(usable, loglik, 0.0).sum(axis=1)

    dmu = mu * (1.0 - mu)
    working = np.where(usable, phi_mat**2 * (_trigamma(a) + _trigamma(b)) * dmu**2, 0.0)
    if design.shape[1] == 1:
        scale = (working * design[:, 0] ** 2).sum(axis=1)
        with np.errstate(divide="ignore"):
            apl = apl - 0.5 * np.where(scale < _LOW, _LOG_LOW, np.log(scale))
    else:
        info = np.einsum("ji,mj,jk->mik", design, working, design)
        _, piv, _ = _batch_cholesky(info)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_piv = np.where(
                (np.abs(piv) < _LOW) | ~np.isfinite(piv), _LOG_LOW, np.log(np.abs(piv))
            )
        apl = apl - 0.5 * log_piv.sum(axis=1)

    apl = np.asarray(apl, dtype=np.float64)
    return (apl, coef) if return_coef else apl


def common_precision(
    y: FloatArray,
    design: FloatArray,
    interval: tuple[float, float] = (1e-4, 1e4),
    tol: float = 1e-5,
) -> float:
    """Common beta precision maximizing the summed Cox-Reid adjusted profile likelihood."""

    def objective(par: float) -> float:
        apl = adjusted_profile_likelihood_beta(par**4, y, design)
        return -float(np.sum(np.asarray(apl)))

    result = minimize_scalar(
        objective,
        bounds=(interval[0] ** 0.25, interval[1] ** 0.25),
        # Pinned: SciPy < 1.10 ignores `bounds` unless method='bounded'.
        method="bounded",
        options={"xatol": tol},
    )
    return float(result.x) ** 4


def _visit_order(n_points: int) -> list[int]:
    """Grid indices ordered outward from the centre, so each fit warm-starts from a neighbour."""
    center = n_points // 2
    left = list(range(center - 1, -1, -1))
    right = list(range(center + 1, n_points))
    order = [center]
    for i in range(max(len(left), len(right))):
        if i < len(right):
            order.append(right[i])
        if i < len(left):
            order.append(left[i])
    return order


def featurewise_precision(
    y: FloatArray,
    design: FloatArray,
    phi_common: float,
    grid_npts: int = 11,
    grid_range: tuple[float, float] = (-6.0, 6.0),
    maxit: int = 50,
    tol: float = 1e-4,
) -> FloatArray:
    """Per-feature beta precision by Cox-Reid APL on a log2 grid around ``phi_common``."""
    n_features = y.shape[0]
    spline_pts: FloatArray = np.linspace(grid_range[0], grid_range[1], grid_npts, dtype=np.float64)

    apl = np.zeros((n_features, grid_npts))
    cache: list[FloatArray | None] = [None] * grid_npts
    for i in _visit_order(grid_npts):
        warm = cache[i - 1] if i > 0 else None
        if warm is None and i + 1 < grid_npts:
            warm = cache[i + 1]
        values, coef = adjusted_profile_likelihood_beta(
            phi_common * 2.0 ** spline_pts[i],
            y,
            design,
            coef_start=warm,
            maxit=maxit,
            tol=tol,
            return_coef=True,
        )
        apl[:, i] = values
        cache[i] = coef

    return np.asarray(phi_common * 2.0 ** maximize_interpolant(spline_pts, apl), dtype=np.float64)

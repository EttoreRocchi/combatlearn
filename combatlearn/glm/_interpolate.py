"""Cubic-spline maximization shared by the GLM dispersion estimators.

The spline uses the Forsythe, Malcolm & Moler (1977) end conditions: the third
derivative at each boundary matches that of the cubic through the four nearest points.
"""

from __future__ import annotations

import numpy as np

from ..core import FloatArray


def _fmm_spline(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Per-row coefficients ``b, c, d`` of the FMM interpolant on shared knots ``x``.

    On segment ``i`` the interpolant is ``y[i] + b[i]*t + c[i]*t**2 + d[i]*t**3``
    with ``t = xi - x[i]``.
    """
    n = x.shape[0]
    m = y.shape[0]
    if n < 2:
        raise ValueError("spline interpolation requires at least two points")

    if n == 2:
        slope = (y[:, 1] - y[:, 0]) / (x[1] - x[0])
        b = np.repeat(slope[:, None], 2, axis=1)
        zeros = np.zeros((m, 2))
        return b, zeros, zeros.copy()

    dx = np.diff(x)
    slope = np.diff(y, axis=1) / dx

    diag = np.zeros(n)
    diag[1 : n - 1] = 2.0 * (dx[:-1] + dx[1:])
    rhs = np.zeros((m, n))
    rhs[:, 1 : n - 1] = np.diff(slope, axis=1)

    # End conditions from the divided differences of the outermost four points.
    diag[0] = -dx[0]
    diag[n - 1] = -dx[n - 2]
    if n > 3:
        lo = rhs[:, 2] / (x[3] - x[1]) - rhs[:, 1] / (x[2] - x[0])
        hi = rhs[:, n - 2] / (x[n - 1] - x[n - 3]) - rhs[:, n - 3] / (x[n - 2] - x[n - 4])
        rhs[:, 0] = lo * dx[0] ** 2 / (x[3] - x[0])
        rhs[:, n - 1] = -hi * dx[n - 2] ** 2 / (x[n - 1] - x[n - 4])

    for i in range(1, n):
        t = dx[i - 1] / diag[i - 1]
        diag[i] -= t * dx[i - 1]
        rhs[:, i] -= t * rhs[:, i - 1]

    rhs[:, n - 1] /= diag[n - 1]
    for i in range(n - 2, -1, -1):
        rhs[:, i] = (rhs[:, i] - dx[i] * rhs[:, i + 1]) / diag[i]

    b = np.empty((m, n))
    d = np.empty((m, n))
    b[:, n - 1] = slope[:, n - 2] + dx[n - 2] * (rhs[:, n - 2] + 2.0 * rhs[:, n - 1])
    b[:, : n - 1] = slope - dx * (rhs[:, 1:] + 2.0 * rhs[:, : n - 1])
    d[:, : n - 1] = (rhs[:, 1:] - rhs[:, : n - 1]) / dx
    d[:, n - 1] = d[:, n - 2]
    return b, 3.0 * rhs, d


def _stationary_point(a: FloatArray, b: FloatArray, c: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Root of ``a*t**2 + b*t + c`` at the cubic's local maximum, and solvability.

    For a quadratic, the ``-sqrt`` root is the maximum for either sign of ``a``.
    """
    solvable = np.ones(a.shape, dtype=bool)
    root = np.zeros(a.shape)

    linear = a == 0.0
    degenerate = linear & (b == 0.0)
    solvable &= ~degenerate

    with np.errstate(divide="ignore", invalid="ignore"):
        root = np.where(linear & ~degenerate, -c / np.where(b == 0.0, 1.0, b), root)

        quadratic = ~linear
        disc = b**2 - 4.0 * a * c
        solvable &= ~(quadratic & (disc < 0.0))
        safe = np.where(quadratic & (disc >= 0.0), disc, 0.0)
        two_a = np.where(quadratic, 2.0 * a, 1.0)
        root = np.where(quadratic, -b / two_a - np.sqrt(safe) / two_a, root)
    return root, solvable


def maximize_interpolant(x: FloatArray, y: FloatArray) -> FloatArray:
    """Locate the maximum of the cubic spline interpolant of each row of ``y``.

    Starts from the grid maximum, then searches its two adjacent segments for
    an interior stationary point that beats it.

    Parameters
    ----------
    x : ndarray of shape (n_points,)
        Strictly increasing grid, shared by every row.
    y : ndarray of shape (n_rows, n_points)
        Function values on the grid.

    Returns
    -------
    ndarray of shape (n_rows,)
        The abscissa of the maximum for each row.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("`y` must be a 2-D array of function values")
    if x.shape[0] != y.shape[1]:
        raise ValueError("`y` must have one column per grid point")

    n_rows, n_points = y.shape
    b, c, d = _fmm_spline(x, y)
    rows = np.arange(n_rows)

    at = np.argmax(y, axis=1)
    best = y[rows, at]
    x_max = x[at]

    for offset in (-1, 0):
        seg = at + offset
        valid = (seg >= 0) & (seg < n_points - 1)
        if not valid.any():
            continue
        sg = np.clip(seg, 0, n_points - 2)
        sb, sc, sd = b[rows, sg], c[rows, sg], d[rows, sg]
        root, solvable = _stationary_point(3.0 * sd, 2.0 * sc, sb)
        width = x[sg + 1] - x[sg]
        inside = valid & solvable & (root > 0.0) & (root < width)
        value = ((sd * root + sc) * root + sb) * root + y[rows, sg]
        better = inside & (value > best)
        x_max = np.where(better, root + x[sg], x_max)
        best = np.where(better, value, best)
    return np.asarray(x_max, dtype=np.float64)

"""Single-unit empirical-Bayes degeneracy: EB must reduce to the MLE, not NaN.

With fewer than two units (features, or PCs for CovBat) the non-parametric prior
used to take the variance of a single value (0 DoF -> NaN). The fix degenerates to
the per-unit MLE (no shrinkage) for both branches.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from utils import simulate_data

from combatlearn import ComBat


def _single_feature_data(seed=0):
    rng = np.random.default_rng(seed)
    n = 150
    batch = pd.Series(rng.choice(list("ABC"), n))
    X = pd.DataFrame(rng.standard_normal((n, 1)))
    X.values[(batch == "A").to_numpy()] += 2.0
    return X, batch


def _kw_for(method, X):
    kw = {}
    if method in {"fortin", "chen", "gam", "covbat_gam"}:
        kw["continuous_covariates"] = pd.DataFrame(
            {"age": np.linspace(0.0, 1.0, len(X))}, index=X.index
        )
    if method == "longitudinal":
        kw["subject_id"] = pd.Series([f"s{i % 50}" for i in range(len(X))], index=X.index)
    return kw


@pytest.mark.parametrize(
    "method", ["johnson", "fortin", "chen", "gam", "covbat_gam", "longitudinal"]
)
def test_nonparametric_single_unit_is_finite(method):
    """No method produces NaN/Inf with a single feature under non-parametric EB."""
    X, batch = _single_feature_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ComBat(
            batch=batch, method=method, parametric=False, **_kw_for(method, X)
        ).fit_transform(X)
    assert np.isfinite(out.values).all()


def test_covbat_nonparametric_single_pc_is_finite():
    """Many features collapsing to one PC (the common CovBat trigger) stays finite."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.standard_normal((n, 10)))
    X.values[:40] += 3.0
    batch = pd.Series(rng.choice(list("ABC"), n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ComBat(
            batch=batch, method="chen", parametric=False, covbat_cov_thresh=1
        ).fit_transform(X)
    assert np.isfinite(out.values).all()


def test_single_feature_param_equals_nonparam_mle():
    """With one unit both branches collapse to the same no-shrinkage MLE."""
    X, batch = _single_feature_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        par = ComBat(batch=batch, method="johnson", parametric=True).fit_transform(X)
        nonpar = ComBat(batch=batch, method="johnson", parametric=False).fit_transform(X)
    np.testing.assert_allclose(par.values, nonpar.values, rtol=1e-10, atol=1e-10)


def test_single_feature_fully_removes_batch_means():
    """MLE (no shrinkage) maps every batch mean to the grand mean."""
    X, batch = _single_feature_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = ComBat(batch=batch, method="johnson", parametric=False).fit_transform(X)
    batch_means = out.groupby(batch.values).mean().values.ravel()
    np.testing.assert_allclose(batch_means, batch_means[0], atol=1e-8)


def test_multifeature_still_runs_eb_not_shortcircuit():
    """With >=2 features the short-circuit must NOT trigger (EB still iterates)."""
    X, batch = simulate_data()  # 25 features
    est = ComBat(batch=batch, method="johnson", parametric=False).fit(X)
    info = est._model._convergence_info
    assert all(c["iterations"] >= 1 for c in info)


def test_covbat_single_pc_warns():
    """CovBat retaining one PC warns and points to fortin / a higher threshold."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame(rng.standard_normal((n, 8)))
    batch = pd.Series(rng.choice(list("AB"), n))
    with pytest.warns(UserWarning, match="single principal component"):
        ComBat(batch=batch, method="chen", covbat_cov_thresh=1).fit(X)


def test_single_feature_warns():
    """A one-feature input warns that EB shrinkage is inactive."""
    X, batch = _single_feature_data()
    with pytest.warns(UserWarning, match="[Oo]nly one feature"):
        ComBat(batch=batch, method="johnson").fit(X)

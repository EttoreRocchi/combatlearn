"""Tests for combatlearn.transductive.TransductiveComBat and the longitudinal deprecation."""

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from utils import simulate_longitudinal_data

from combatlearn import ComBat
from combatlearn.transductive import TransductiveComBat


def _cohort():
    X, batch, subject, time = simulate_longitudinal_data()
    return X, {"batch": batch, "subject_id": subject, "time_covariate": time}


def test_fit_transform_matches_combat_longitudinal():
    """fit_transform reproduces ComBat(method='longitudinal').fit_transform exactly."""
    X, kwargs = _cohort()
    transductive = TransductiveComBat(**kwargs).fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        inductive = ComBat(method="longitudinal", **kwargs).fit_transform(X)
    np.testing.assert_allclose(transductive.values, inductive.values, rtol=1e-12, atol=1e-12)


def test_separate_transform_raises():
    """transform on held-out data is not supported and raises a clear error."""
    X, kwargs = _cohort()
    est = TransductiveComBat(**kwargs).fit(X)
    with pytest.raises(NotImplementedError, match="fit_transform-only"):
        est.transform(X)


def test_longitudinal_deprecation_warning_fires():
    """ComBat(method='longitudinal') warns and points to TransductiveComBat."""
    X, kwargs = _cohort()
    with pytest.warns(DeprecationWarning, match="TransductiveComBat"):
        ComBat(method="longitudinal", **kwargs).fit(X)


def test_fit_transform_shape_and_finite():
    """Output preserves shape and introduces no NaN/Inf."""
    X, kwargs = _cohort()
    out = TransductiveComBat(**kwargs).fit_transform(X)
    assert out.shape == X.shape
    assert np.isfinite(out.values).all()


def test_longcombat_alias_supported():
    """The 'longcombat' alias resolves to the longitudinal engine."""
    X, kwargs = _cohort()
    out = TransductiveComBat(method="longcombat", **kwargs).fit_transform(X)
    assert np.isfinite(out.values).all()


@pytest.mark.parametrize("method", ["seq", "met", "fortin"])
def test_unsupported_method_raises(method):
    """Only the longitudinal engine is available for now."""
    X, kwargs = _cohort()
    with pytest.raises(ValueError, match="not available in TransductiveComBat"):
        TransductiveComBat(method=method, **kwargs).fit(X)


def test_clone_get_params():
    """The estimator clones and exposes its constructor params."""
    X, kwargs = _cohort()
    est = TransductiveComBat(**kwargs)
    cloned = clone(est)
    assert cloned.get_params()["method"] == "longitudinal"
    out = cloned.fit_transform(X)
    assert np.isfinite(out.values).all()

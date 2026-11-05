"""Tests for combatlearn.transductive.TransductiveComBat and the longitudinal deprecation."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from utils import simulate_longitudinal_data

from combatlearn import ComBat, GLMComBat
from combatlearn.transductive import TransductiveComBat


def _cohort():
    X, batch, subject, time = simulate_longitudinal_data()
    return X, {
        "method": "longitudinal",
        "batch": batch,
        "subject_id": subject,
        "time_covariate": time,
    }


def _counts(n_samples=24, n_features=30, seed=0):
    """Counts with a multiplicative batch effect and a condition-of-interest effect."""
    rng = np.random.default_rng(seed)
    batch = pd.Series(np.repeat(["A", "B", "C"], n_samples // 3), name="batch")
    condition = pd.Series(np.tile(["ctrl", "case"], n_samples // 2), name="condition")
    base = rng.gamma(5, 60, size=(n_features, 1))
    batch_effect = np.exp(rng.normal(0, 0.4, size=(n_features, 3)))
    cond_effect = np.exp(rng.normal(0, 0.3, size=(n_features, 2)))
    mu = (
        base
        * batch_effect[:, pd.Categorical(batch).codes]
        * cond_effect[:, pd.Categorical(condition).codes]
    )
    counts = rng.negative_binomial(5, 5 / (5 + mu)).astype(np.int64)
    return pd.DataFrame(counts.T), batch, condition


def test_fit_transform_matches_combat_longitudinal():
    """fit_transform reproduces ComBat(method='longitudinal').fit_transform exactly."""
    X, kwargs = _cohort()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        transductive = TransductiveComBat(**kwargs).fit_transform(X)
        inductive = ComBat(**kwargs).fit_transform(X)
    np.testing.assert_allclose(transductive.values, inductive.values, rtol=1e-12, atol=1e-12)


def test_separate_transform_raises():
    """transform on held-out data is not supported and raises a clear error."""
    X, kwargs = _cohort()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        est = TransductiveComBat(**kwargs).fit(X)
    with pytest.raises(NotImplementedError, match="fit_transform-only"):
        est.transform(X)


def test_longitudinal_deprecation_warning_fires():
    """ComBat(method='longitudinal') warns and points to LongitudinalComBat."""
    X, kwargs = _cohort()
    with pytest.warns(DeprecationWarning, match="LongitudinalComBat"):
        ComBat(**kwargs).fit(X)


def test_transductive_longitudinal_is_deprecated():
    """method='longitudinal' raises a DeprecationWarning calling it redundant."""
    X, kwargs = _cohort()
    with pytest.warns(DeprecationWarning, match="redundant"):
        TransductiveComBat(**kwargs).fit(X)


def test_fit_transform_shape_and_finite():
    """Output preserves shape and introduces no NaN/Inf."""
    X, kwargs = _cohort()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = TransductiveComBat(**kwargs).fit_transform(X)
    assert out.shape == X.shape
    assert np.isfinite(out.values).all()


def test_longcombat_alias_supported():
    """The 'longcombat' alias resolves to the longitudinal engine."""
    X, kwargs = _cohort()
    kwargs["method"] = "longcombat"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = TransductiveComBat(**kwargs).fit_transform(X)
    assert np.isfinite(out.values).all()


@pytest.mark.parametrize("method", ["fortin", "chen", "gam", "nonsense"])
def test_unsupported_method_raises(method):
    """Inductive-only or unknown methods are rejected at fit."""
    X, kwargs = _cohort()
    kwargs["method"] = method
    with pytest.raises(ValueError, match="not available in TransductiveComBat"):
        TransductiveComBat(**kwargs).fit(X)


def test_clone_get_params():
    """The estimator clones and exposes its constructor params."""
    X, batch, condition = _counts()
    est = TransductiveComBat(batch=batch, method="seq", condition=condition)
    cloned = clone(est)
    assert cloned.get_params()["method"] == "seq"
    out = cloned.fit_transform(X)
    assert np.isfinite(out.values).all()


def test_method_defaults_to_the_deprecated_longitudinal_engine():
    """Without ``method`` the 2.3.0 default still runs and warns toward LongitudinalComBat."""
    X, kwargs = _cohort()
    kwargs.pop("method")
    with pytest.warns(DeprecationWarning, match="LongitudinalComBat"):
        out = TransductiveComBat(**kwargs).fit_transform(X)
    assert np.isfinite(out.values).all()


@pytest.mark.parametrize("method", ["seq", "met"])
def test_glm_engines_require_a_condition(method):
    """'seq' and 'met' without ``condition`` are rejected at fit."""
    X, batch, _ = _counts()
    if method == "met":
        X = pd.DataFrame(np.clip(X.values / (X.values.max() + 1.0), 1e-3, 1 - 1e-3))
    with pytest.raises(ValueError, match="requires `condition`"):
        TransductiveComBat(batch=batch, method=method).fit(X)


def test_seq_protects_the_condition():
    """Protecting the condition preserves more of its signal than ignoring it."""
    X, batch, condition = _counts()
    protected = TransductiveComBat(batch=batch, method="seq", condition=condition).fit_transform(X)
    ignored = GLMComBat(batch=batch, method="seq").fit_transform(X)

    def condition_signal(frame):
        values = np.log1p(frame.values.astype(float))
        groups = [values[np.asarray(condition) == lvl] for lvl in ("ctrl", "case")]
        return float(np.abs(groups[0].mean(axis=0) - groups[1].mean(axis=0)).mean())

    assert condition_signal(protected) >= condition_signal(ignored)


def test_met_condition_output_in_unit_interval():
    """ComBat-met output stays strictly inside (0, 1)."""
    rng = np.random.default_rng(4)
    X, batch, condition = _counts()
    values = pd.DataFrame(rng.uniform(0.05, 0.95, size=X.shape))
    out = TransductiveComBat(batch=batch, method="met", condition=condition).fit_transform(values)
    assert ((out.values > 0.0) & (out.values < 1.0)).all()


@pytest.mark.parametrize("method", ["seq", "met"])
def test_condition_is_computationally_a_covariate(method):
    """A protected condition gives output bit-identical to the same column as a covariate."""
    X, batch, condition = _counts(seed=6)
    if method == "met":
        X = X.div(X.values.max() + 1.0)

    protected = TransductiveComBat(batch=batch, method=method, condition=condition).fit_transform(X)
    as_covariate = GLMComBat(
        batch=batch, method=method, discrete_covariates=condition
    ).fit_transform(X)
    np.testing.assert_array_equal(protected.values, as_covariate.values)


def test_glm_transform_raises_with_condition():
    """A fitted condition-protected model refuses to transform held-out data."""
    X, batch, condition = _counts()
    est = TransductiveComBat(batch=batch, method="seq", condition=condition).fit(X)
    with pytest.raises(NotImplementedError, match="fit_transform-only"):
        est.transform(X)


def test_condition_ignored_for_longitudinal():
    """A condition passed to the longitudinal engine warns that it is ignored."""
    X, kwargs = _cohort()
    kwargs["condition"] = pd.Series(np.tile(["a", "b"], len(X) // 2), index=X.index)
    with (
        pytest.warns(UserWarning, match="only used by the 'seq' and 'met' engines"),
        pytest.warns(DeprecationWarning, match="is deprecated"),
    ):
        TransductiveComBat(**kwargs).fit(X)


def test_combat_seq_alias_resolves():
    """The 'combat-seq' alias resolves to the 'seq' engine."""
    X, batch, condition = _counts()
    out = TransductiveComBat(batch=batch, method="combat-seq", condition=condition).fit_transform(X)
    assert out.shape == X.shape

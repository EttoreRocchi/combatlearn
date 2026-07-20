"""Property-based tests (Hypothesis) for the ComBat estimators.

These complement the example-based suites: instead of fixed fixtures they
generate many valid configurations - varying dimensions, batch composition,
covariates, and hyper-parameters - and assert the invariants that must hold for
*every* input:

* shape, index, and column names are preserved,
* no NaN/Inf is introduced,
* the correction is deterministic (same input -> same output),
* ``fit(X).transform(X)`` equals ``fit_transform(X)`` for the inductive estimators,
* reference-batch samples come out untouched,
* the ``NestedComBat`` order is always a permutation of the batch variables,
* ``TransductiveComBat`` is ``fit_transform``-only (``transform`` always raises).

The generators only ever produce *valid* data (every batch level has at least a
few samples), so any exception raised inside ``fit``/``transform`` is a genuine
failure, not a bad-input artefact. Benign fit-time warnings (imbalance,
near-zero variance, rank-deficiency on small random draws) are silenced so the
tests stay focused on the invariants above.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from combatlearn import ComBat, NestedComBat
from combatlearn.transductive import TransductiveComBat

_SETTINGS = settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow],
)

_SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


def _assert_core_invariants(X: pd.DataFrame, out: pd.DataFrame) -> None:
    """Every harmonizer must preserve shape/index/columns and stay finite."""
    assert out.shape == X.shape
    assert list(out.columns) == list(X.columns)
    assert out.index.equals(X.index)
    assert np.isfinite(out.values).all()


@st.composite
def combat_data(draw, *, with_covariates: bool = False):
    """A valid single-batch ComBat scenario.

    Draws the number of batches, per-batch sample counts (each comfortably above
    one, so no batch is degenerate), feature count, and an additive per-batch
    shift, then builds ``(X, batch, discrete, continuous)``. Covariates are
    generated independently of the batch labels (so they are not collinear) and,
    when requested, an age-driven signal is added for the covariate model to fit.
    """
    n_batches = draw(st.integers(min_value=2, max_value=4))
    counts = draw(
        st.lists(
            st.integers(min_value=4, max_value=12),
            min_size=n_batches,
            max_size=n_batches,
        )
    )
    n_features = draw(st.integers(min_value=2, max_value=6))
    shift = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    rng = np.random.default_rng(draw(_SEEDS))

    levels = [chr(ord("A") + i) for i in range(n_batches)]
    labels = np.concatenate([np.full(c, lv) for lv, c in zip(levels, counts, strict=True)])
    labels = labels[rng.permutation(labels.size)]
    n = labels.size

    center = (n_batches - 1) / 2.0
    shift_map = {lv: (i - center) * shift for i, lv in enumerate(levels)}
    values = (
        rng.standard_normal((n, n_features)) + np.array([shift_map[lv] for lv in labels])[:, None]
    )

    X = pd.DataFrame(values, columns=[f"f{i}" for i in range(n_features)])
    batch = pd.Series(labels, index=X.index, name="batch")

    if not with_covariates:
        return X, batch, None, None

    age = rng.uniform(20.0, 80.0, n)
    X = X + (age[:, None] - 50.0) * 0.03
    disc = pd.DataFrame({"grp": rng.choice(["p", "q"], n)}, index=X.index)
    cont = pd.DataFrame({"age": age}, index=X.index)
    return X, batch, disc, cont


def _make_combat(method, batch, disc, cont, **extra):
    """Build a ``ComBat`` for ``method``, passing covariates only when they apply."""
    kwargs = dict(batch=batch, method=method, **extra)
    if method != "johnson":
        kwargs["discrete_covariates"] = disc
        kwargs["continuous_covariates"] = cont
    if method in ("gam", "covbat_gam"):
        # Small random draws can't support the default df=10 basis; shrink it so
        # the spline stays well-conditioned regardless of the drawn sample size.
        kwargs.setdefault("spline_df", 5)
    return ComBat(**kwargs)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen", "gam", "covbat_gam"])
@_SETTINGS
@given(
    scenario=combat_data(with_covariates=True), parametric=st.booleans(), mean_only=st.booleans()
)
def test_property_combat_invariants(method, scenario, parametric, mean_only):
    """Every method/flag combination preserves structure and stays finite."""
    X, batch, disc, cont = scenario
    est = _make_combat(method, batch, disc, cont, parametric=parametric, mean_only=mean_only)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = est.fit_transform(X)
    _assert_core_invariants(X, out)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen", "gam", "covbat_gam"])
@_SETTINGS
@given(scenario=combat_data(with_covariates=True))
def test_property_combat_deterministic(method, scenario):
    """Fitting the same data twice yields bit-identical output."""
    X, batch, disc, cont = scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = _make_combat(method, batch, disc, cont).fit_transform(X)
        b = _make_combat(method, batch, disc, cont).fit_transform(X)
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen", "gam", "covbat_gam"])
@_SETTINGS
@given(scenario=combat_data(with_covariates=True))
def test_property_fit_transform_matches_split(method, scenario):
    """``fit(X).transform(X)`` equals ``fit_transform(X)`` (inductive contract)."""
    X, batch, disc, cont = scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        split = _make_combat(method, batch, disc, cont).fit(X).transform(X)
        joint = _make_combat(method, batch, disc, cont).fit_transform(X)
    np.testing.assert_allclose(split.values, joint.values, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
@_SETTINGS
@given(scenario=combat_data(with_covariates=True))
def test_property_reference_batch_samples_untouched(method, scenario):
    """Samples in the reference batch are returned numerically unchanged."""
    X, batch, disc, cont = scenario
    ref = batch.iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _make_combat(method, batch, disc, cont, reference_batch=ref).fit_transform(X)
    mask = (batch == ref).to_numpy()
    np.testing.assert_allclose(out.values[mask], X.values[mask], rtol=0, atol=1e-9)


def _balanced_labels(rng, n, levels, min_count=3):
    """Labels with at least ``min_count`` samples per level, then shuffled."""
    base = np.repeat(np.asarray(levels), min_count)
    extra = rng.choice(levels, max(0, n - base.size))
    labels = np.concatenate([base, extra])[:n]
    return labels[rng.permutation(n)]


@st.composite
def nested_data(draw, *, n_vars: int | None = None):
    """A valid multi-batch scenario for ``NestedComBat``.

    Builds a batch DataFrame with ``n_vars`` variables (site/scanner/protocol),
    each level guaranteed a few samples, over a shared age signal to protect.
    """
    n = draw(st.integers(min_value=40, max_value=80))
    n_features = draw(st.integers(min_value=3, max_value=6))
    k = n_vars if n_vars is not None else draw(st.integers(min_value=2, max_value=3))
    rng = np.random.default_rng(draw(_SEEDS))

    var_levels = [["S1", "S2"], ["G1", "G2", "G3"], ["P1", "P2"]][:k]
    names = ["site", "scanner", "protocol"][:k]

    total_shift = np.zeros((n, n_features))
    batch_cols = {}
    for name, levels in zip(names, var_levels, strict=True):
        assign = _balanced_labels(rng, n, levels)
        batch_cols[name] = assign
        shift_map = {lv: rng.uniform(-2.5, 2.5) for lv in levels}
        per_sample = np.array([shift_map[a] for a in assign])[:, None]
        total_shift += per_sample * rng.uniform(0.5, 1.5, n_features)

    age = rng.uniform(20.0, 80.0, n)
    effect = np.outer((age - 50.0) / 30.0, rng.uniform(0.5, 1.5, n_features))
    values = effect + rng.standard_normal((n, n_features)) + total_shift

    X = pd.DataFrame(values, columns=[f"f{i}" for i in range(n_features)])
    idx = X.index
    batch = pd.DataFrame({k_: pd.Series(v, index=idx) for k_, v in batch_cols.items()})
    disc = pd.DataFrame({"sex": rng.choice(["M", "F"], n)}, index=idx)
    cont = pd.DataFrame({"age": age}, index=idx)
    return X, batch, disc, cont


@settings(deadline=None, max_examples=20, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=nested_data())
def test_property_nested_invariants(scenario):
    """NestedComBat preserves structure and covers every batch variable once."""
    X, batch, disc, cont = scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = NestedComBat(
            batch=batch,
            discrete_covariates=disc,
            continuous_covariates=cont,
            optimize_order=False,
        ).fit(X)
        out = est.transform(X)
    _assert_core_invariants(X, out)
    assert sorted(est.order_) == sorted(batch.columns)


@settings(deadline=None, max_examples=15, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=nested_data(n_vars=2))
def test_property_nested_optimized_order_is_deterministic_permutation(scenario):
    """The OPNested search returns a reproducible permutation of the batch vars."""
    X, batch, disc, cont = scenario
    common = {"batch": batch, "discrete_covariates": disc, "continuous_covariates": cont}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order1 = NestedComBat(**common, optimize_order=True).fit(X).order_
        order2 = NestedComBat(**common, optimize_order=True).fit(X).order_
    assert sorted(order1) == sorted(batch.columns)
    assert order1 == order2


@st.composite
def longitudinal_data(draw):
    """Repeated-measures data: each subject observed across batches over time."""
    n_subjects = draw(st.integers(min_value=6, max_value=18))
    n_times = draw(st.integers(min_value=2, max_value=4))
    n_features = draw(st.integers(min_value=3, max_value=6))
    rng = np.random.default_rng(draw(_SEEDS))

    batch_levels = ["A", "B", "C"]
    shifts = {"A": 3.0, "B": -3.0, "C": 1.0}
    subject_re = rng.normal(0.0, 1.5, size=(n_subjects, n_features))

    rows, subjects, times, batches = [], [], [], []
    for s in range(n_subjects):
        for t in range(n_times):
            b = batch_levels[(s + t) % len(batch_levels)]
            rows.append(rng.standard_normal(n_features) + subject_re[s] + shifts[b] + 0.5 * t)
            subjects.append(f"subj_{s}")
            times.append(float(t))
            batches.append(b)

    X = pd.DataFrame(np.asarray(rows), columns=[f"f{i}" for i in range(n_features)])
    idx = X.index
    batch = pd.Series(batches, index=idx, name="batch")
    subject = pd.Series(subjects, index=idx, name="subject")
    time = pd.Series(times, index=idx, name="time")
    return X, batch, subject, time


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=longitudinal_data())
def test_property_transductive_fit_transform_invariants(scenario):
    """Whole-cohort longitudinal harmonization preserves structure and stays finite."""
    X, batch, subject, time = scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = TransductiveComBat(
            batch=batch,
            subject_id=subject,
            time_covariate=time,
            method="longitudinal",
        ).fit_transform(X)
    _assert_core_invariants(X, out)


@settings(deadline=None, max_examples=25, suppress_health_check=[HealthCheck.too_slow])
@given(scenario=longitudinal_data())
def test_property_transductive_transform_always_raises(scenario):
    """``transform`` on held-out data is unsupported for any fitted cohort."""
    X, batch, subject, time = scenario
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = TransductiveComBat(batch=batch, subject_id=subject, time_covariate=time).fit(X)
    with pytest.raises(NotImplementedError):
        est.transform(X)

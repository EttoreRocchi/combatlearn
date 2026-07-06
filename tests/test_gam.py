"""Tests for ComBat-GAM (``method='gam'`` / ``method='covbat_gam'``)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import simulate_gam_data

from combatlearn import ComBat
from combatlearn.core import ComBatModel


def _recovery_mse(corrected, oracle):
    """Mean squared error to the batch-free signal, after removing the global offset."""
    c = corrected - corrected.mean(axis=0)
    o = oracle.values - oracle.values.mean(axis=0)
    return float(((c - o) ** 2).mean())


def _batch_var(X, batch):
    grand = X.mean(axis=0)
    ss_total = ((X - grand) ** 2).sum()
    ss_between = 0.0
    for lvl in np.unique(batch):
        mask = batch == lvl
        ss_between += mask.sum() * ((X[mask].mean(axis=0) - grand) ** 2).sum()
    return ss_between / ss_total


@pytest.mark.parametrize("df,degree", [(10, 3), (8, 3), (6, 2), (12, 3), (7, 2)])
def test_spline_basis_matches_statsmodels(df, degree):
    """The scipy B-spline basis (drop-first column) reproduces statsmodels exactly."""
    BSplines = pytest.importorskip("statsmodels.gam.smooth_basis").BSplines
    rng = np.random.default_rng(0)
    x = rng.uniform(20.0, 80.0, 200)

    model = ComBatModel(method="gam", spline_df=df, spline_degree=degree)
    knots = model._spline_knots(x, float(x.min()), float(x.max()))
    ours = model._spline_basis(x, knots)
    reference = BSplines(x, df=df, degree=degree, include_intercept=False).basis

    assert ours.shape == reference.shape == (200, df - 1)
    np.testing.assert_allclose(ours, reference, rtol=1e-10, atol=1e-10)


def test_degree1_linear_spline_basis_is_valid():
    """Degree-1 (piecewise-linear) basis builds a finite, full-rank df-1 design.

    statsmodels cannot construct a degree-1 BSplines basis (it requires a 2nd
    derivative for its penalty), so this is checked without a parity reference.
    """
    rng = np.random.default_rng(0)
    x = rng.uniform(20.0, 80.0, 200)
    model = ComBatModel(method="gam", spline_df=5, spline_degree=1)
    knots = model._spline_knots(x, float(x.min()), float(x.max()))
    basis = model._spline_basis(x, knots)
    assert basis.shape == (200, 4)
    assert np.isfinite(basis).all()
    assert np.linalg.matrix_rank(basis) == 4


def test_fitted_knots_match_statsmodels_quantile_rule():
    """Stored knots = degree+1 boundary repeats + (df-degree-1) interior quantiles."""
    pytest.importorskip("statsmodels")
    X, batch, cont, _ = simulate_gam_data(random_state=1)
    model = ComBatModel(method="gam").fit(X, batch=batch, continuous_covariates=cont)
    knots = model._smooth_knots["age"]
    age = cont["age"].to_numpy()
    n_interior = 10 - 3 - 1
    expected_interior = np.quantile(age, np.linspace(0, 1, n_interior + 2)[1:-1])
    assert knots[:4] == pytest.approx(age.min())
    assert knots[-4:] == pytest.approx(age.max())
    np.testing.assert_allclose(knots[4:-4], expected_interior, rtol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gam_recovers_nonlinear_effect_better_than_fortin(seed):
    """Under nonlinear age-batch confounding, gam recovers the signal far better."""
    X, batch, cont, oracle = simulate_gam_data(confound=True, nonlinear=True, random_state=seed)
    fortin = ComBat(batch=batch, continuous_covariates=cont, method="fortin").fit_transform(X)
    gam = ComBat(batch=batch, continuous_covariates=cont, method="gam").fit_transform(X)
    mse_fortin = _recovery_mse(fortin.values, oracle)
    mse_gam = _recovery_mse(gam.values, oracle)
    assert mse_gam < mse_fortin
    assert mse_gam < 1.0  # gam essentially recovers the batch-free signal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_covbat_gam_engages_spline_path(seed):
    """covbat_gam runs the spline mean model, so it is finite and differs from linear chen.

    (covbat_gam is not compared on signal-recovery MSE: CovBat's covariance step
    legitimately moves data off the additive oracle, washing out the mean-model gain
    on that metric. The spline mean-model benefit itself is covered by the ``gam`` test.)
    """
    X, batch, cont, _ = simulate_gam_data(confound=True, nonlinear=True, random_state=seed)
    chen = ComBat(batch=batch, continuous_covariates=cont, method="chen").fit_transform(X)
    covbat_gam = ComBat(batch=batch, continuous_covariates=cont, method="covbat_gam").fit_transform(
        X
    )
    assert np.isfinite(covbat_gam.values).all()
    assert not np.allclose(covbat_gam.values, chen.values)


def test_gam_matches_fortin_on_linear_unconfounded_effect():
    """When the true effect is linear and unconfounded, splines reduce to ~fortin."""
    X, batch, cont, _ = simulate_gam_data(confound=False, nonlinear=False, random_state=0)
    fortin = ComBat(batch=batch, continuous_covariates=cont, method="fortin").fit_transform(X)
    gam = ComBat(batch=batch, continuous_covariates=cont, method="gam").fit_transform(X)
    corr = np.corrcoef(fortin.values.ravel(), gam.values.ravel())[0, 1]
    assert corr > 0.999


def test_gam_design_is_full_rank():
    """The [batch | spline] design must stay full rank (drop-first identifiability)."""
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    model = ComBatModel(method="gam").fit(X, batch=batch, continuous_covariates=cont)
    assert np.isfinite(model._design_cond)
    assert model._design_cond < 1e10


@pytest.mark.parametrize("method", ["gam", "covbat_gam"])
@pytest.mark.parametrize("parametric", [True, False])
@pytest.mark.parametrize("mean_only", [True, False])
def test_no_nan_or_inf(method, parametric, mean_only):
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    out = ComBat(
        batch=batch,
        continuous_covariates=cont,
        method=method,
        parametric=parametric,
        mean_only=mean_only,
    ).fit_transform(X)
    assert np.isfinite(out.values).all()


@pytest.mark.parametrize("method", ["gam", "covbat_gam"])
def test_dtypes_preserved(method):
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    out = ComBat(batch=batch, continuous_covariates=cont, method=method).fit_transform(X)
    assert all(np.issubdtype(dt, np.floating) for dt in out.dtypes)


def test_split_fit_transform_matches_fit_transform():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    a = ComBat(batch=batch, continuous_covariates=cont, method="gam").fit(X).transform(X)
    b = ComBat(batch=batch, continuous_covariates=cont, method="gam").fit_transform(X)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-10, atol=1e-10)


def test_reference_batch_rows_unchanged():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    est = ComBat(batch=batch, continuous_covariates=cont, method="gam", reference_batch="A")
    out = est.fit_transform(X)
    ref_mask = (batch == "A").to_numpy()
    np.testing.assert_allclose(out.values[ref_mask], X.values[ref_mask], rtol=1e-8, atol=1e-8)


def test_out_of_range_transform_clamps_and_is_finite():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    model = ComBatModel(method="gam").fit(X, batch=batch, continuous_covariates=cont)

    cont_oor = cont.copy()
    cont_oor.iloc[:5, 0] = 500.0  # far above the training range
    cont_oor.iloc[5:10, 0] = -100.0  # far below
    out = model.transform(X, batch=batch, continuous_covariates=cont_oor)
    assert np.isfinite(out.values).all()

    # In-range rows are unaffected by the out-of-range rows.
    base = model.transform(X, batch=batch, continuous_covariates=cont)
    np.testing.assert_allclose(out.values[10:], base.values[10:], rtol=1e-8, atol=1e-8)


def test_gam_requires_continuous_covariates():
    X, batch, _, _ = simulate_gam_data(random_state=0)
    with pytest.raises(ValueError, match="requires at least one continuous"):
        ComBat(batch=batch, method="gam").fit(X)


def test_empty_smooth_terms_list_raises():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    with pytest.raises(ValueError, match="empty list"):
        ComBat(batch=batch, continuous_covariates=cont, method="gam", smooth_terms=[]).fit(X)


def test_smooth_term_too_few_distinct_values_raises():
    X, batch, _, _ = simulate_gam_data(random_state=0)
    binary = pd.DataFrame({"flag": np.tile([0.0, 1.0], len(X) // 2)}, index=X.index)
    with pytest.raises(ValueError, match="distinct value"):
        ComBat(batch=batch, continuous_covariates=binary, method="gam").fit(X)


def test_unknown_smooth_term_raises():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    with pytest.raises(ValueError, match="not found"):
        ComBat(batch=batch, continuous_covariates=cont, method="gam", smooth_terms=["nope"]).fit(X)


@pytest.mark.parametrize("kwargs", [{"spline_degree": 0}, {"spline_df": 3, "spline_degree": 3}])
def test_invalid_spline_params_raise(kwargs):
    with pytest.raises(ValueError):
        ComBatModel(method="gam", **kwargs)


def test_smooth_terms_subset_leaves_others_linear():
    X, batch, _, _ = simulate_gam_data(random_state=0)
    rng = np.random.default_rng(0)
    cont = pd.DataFrame(
        {"age": rng.uniform(20, 80, len(X)), "score": rng.uniform(0, 100, len(X))},
        index=X.index,
    )
    model = ComBatModel(method="gam", smooth_terms=["age"]).fit(
        X, batch=batch, continuous_covariates=cont
    )
    assert model._smooth_cols == ["age"]
    assert any(c.startswith("age__spline") for c in model._nonbatch_columns)
    assert "score" in model._nonbatch_columns  # kept linear
    assert not any(c.startswith("score__spline") for c in model._nonbatch_columns)


def test_smooth_term_bounds_tuple_and_dict_run():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    for bounds in [(0.0, 120.0), {"age": (0.0, 120.0)}]:
        out = ComBat(
            batch=batch, continuous_covariates=cont, method="gam", smooth_term_bounds=bounds
        ).fit_transform(X)
        assert np.isfinite(out.values).all()


def test_smooth_term_bounds_inside_data_raises():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    with pytest.raises(ValueError, match="do not contain the training range"):
        ComBat(
            batch=batch,
            continuous_covariates=cont,
            method="gam",
            smooth_term_bounds=(40.0, 60.0),  # narrower than the [20, 80] data range
        ).fit(X)


@pytest.mark.parametrize("alias,canonical", [("combat_gam", "gam"), ("covbatgam", "covbat_gam")])
def test_gam_aliases_match_canonical(alias, canonical):
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    out_alias = ComBat(batch=batch, continuous_covariates=cont, method=alias).fit_transform(X)
    out_canon = ComBat(batch=batch, continuous_covariates=cont, method=canonical).fit_transform(X)
    np.testing.assert_allclose(out_alias.values, out_canon.values, rtol=1e-10, atol=1e-10)


def test_gam_in_pipeline_and_clone():
    X, batch, cont, _ = simulate_gam_data(random_state=0)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("combat", ComBat(batch=batch, continuous_covariates=cont, method="gam")),
        ]
    )
    out1 = pipe.fit_transform(X)
    out2 = clone(pipe).fit_transform(X)
    np.testing.assert_allclose(out1, out2, rtol=1e-8, atol=1e-8)


def test_gam_no_spurious_warnings_on_clean_data():
    """A well-specified gam fit on clean data should not warn."""
    X, batch, cont, _ = simulate_gam_data(confound=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ComBat(batch=batch, continuous_covariates=cont, method="gam").fit_transform(X)

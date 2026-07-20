"""Tests for NestedComBat (Nested / OPNested / GMM ComBat)."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import simulate_nested_data

from combatlearn import NestedComBat
from combatlearn.core import ComBatModel


def test_shape_index_columns_preserved():
    """Output keeps shape, index, and column names."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    X.index = pd.Index([f"obs_{i}" for i in range(len(X))])
    batch.index = X.index
    disc.index = X.index
    cont.index = X.index

    out = NestedComBat(
        batch=batch, discrete_covariates=disc, continuous_covariates=cont
    ).fit_transform(X)
    assert out.shape == X.shape
    assert out.index.equals(X.index)
    assert list(out.columns) == list(X.columns)


@pytest.mark.parametrize("method", ["fortin", "chen", "gam", "covbat_gam"])
def test_engines_run_without_nan(method):
    """Every supported per-step engine runs and introduces no NaN/Inf."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    out = NestedComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method=method,
    ).fit_transform(X)
    assert np.isfinite(out.values).all()


def test_per_variable_batch_variance_reduction():
    """Each batch variable's variance fraction drops after correction."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    est = NestedComBat(batch=batch, discrete_covariates=disc, continuous_covariates=cont).fit(X)
    assert set(est.order_) == set(batch.columns)
    for name in batch.columns:
        assert est.batch_var_after_[name] < est.batch_var_before_[name]


def test_chosen_order_is_deterministic():
    """The OPNested order search is reproducible across fits."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    order1 = (
        NestedComBat(batch=batch, discrete_covariates=disc, continuous_covariates=cont)
        .fit(X)
        .order_
    )
    order2 = (
        NestedComBat(batch=batch, discrete_covariates=disc, continuous_covariates=cont)
        .fit(X)
        .order_
    )
    assert order1 == order2
    assert sorted(order1) == sorted(batch.columns)


def test_optimize_order_false_keeps_given_order():
    """Without order optimization the batch variables are used as given."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    est = NestedComBat(batch=batch, optimize_order=False).fit(X)
    assert est.order_ == list(batch.columns)
    assert est.used_greedy_ is False


def test_guardrail_triggers_greedy_and_warns_past_cap():
    """More batch variables than the cap falls back to greedy with a warning."""
    X, batch, _, _ = simulate_nested_data(n_vars=3, random_state=0)
    with pytest.warns(UserWarning, match="greedy forward selection"):
        est = NestedComBat(batch=batch, max_exhaustive_vars=2).fit(X)
    assert est.used_greedy_ is True
    assert sorted(est.order_) == sorted(batch.columns)


def test_large_exhaustive_search_warns_fit_count():
    """A large exhaustive order search (k >= 4) warns up front about the fit count."""
    X, batch, _, _ = simulate_nested_data(n_samples=120, n_features=8, random_state=0)
    rng = np.random.default_rng(1)
    batch = batch.copy()
    batch["extra"] = pd.Series(rng.choice(["E1", "E2"], size=len(X)), index=X.index)
    with pytest.warns(UserWarning, match="ComBat models"):
        est = NestedComBat(batch=batch).fit(X)
    assert est.used_greedy_ is False
    assert sorted(est.order_) == sorted(batch.columns)


@pytest.mark.parametrize("role", ["batch", "covariate"])
def test_gmm_roles_run(role):
    """Both +GMM (batch) and -GMM (covariate) run and produce finite output."""
    X, batch, disc, cont = simulate_nested_data(latent_split=True, random_state=0)
    est = NestedComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        gmm=role,
        random_state=0,
    ).fit(X)
    assert est._gmm_grouping is not None
    out = est.transform(X)
    assert np.isfinite(out.values).all()


def test_gmm_batch_and_covariate_differ():
    """+GMM (harmonize the split) and -GMM (protect the split) give different results."""
    X, batch, disc, cont = simulate_nested_data(latent_split=True, random_state=0)
    common = {
        "batch": batch,
        "discrete_covariates": disc,
        "continuous_covariates": cont,
        "random_state": 0,
    }
    as_batch = NestedComBat(**common, gmm="batch").fit_transform(X)
    as_cov = NestedComBat(**common, gmm="covariate").fit_transform(X)
    assert not np.allclose(as_batch.values, as_cov.values)


def test_gmm_batch_adds_extra_variable_to_order():
    """gmm='batch' adds a 'GMM' batch variable that joins the harmonization order."""
    X, batch, _, _ = simulate_nested_data(latent_split=True, random_state=0)
    est = NestedComBat(batch=batch, gmm="batch", random_state=0).fit(X)
    assert "GMM" in est.order_
    assert set(est.order_) == set(batch.columns) | {"GMM"}


def test_fit_train_transform_test():
    """Fit on a training split, transform a held-out split (inductive)."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    train, test = X.index[:180], X.index[180:]
    est = NestedComBat(batch=batch, discrete_covariates=disc, continuous_covariates=cont).fit(
        X.loc[train]
    )
    out = est.transform(X.loc[test])
    assert out.shape == X.loc[test].shape
    assert out.index.equals(test)
    assert np.isfinite(out.values).all()


def test_split_fit_transform_matches_fit_transform():
    """Separate fit + transform equals fit_transform on the same cohort."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    common = {"batch": batch, "discrete_covariates": disc, "continuous_covariates": cont}
    a = NestedComBat(**common).fit(X).transform(X)
    b = NestedComBat(**common).fit_transform(X)
    np.testing.assert_allclose(a.values, b.values, rtol=1e-10, atol=1e-10)


def test_unseen_batch_level_raises():
    """A batch level present only at transform raises a clear error."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    batch = batch.copy()
    batch.iloc[-20:, 0] = "S_unseen"
    train, test = X.index[:-20], X.index[-20:]
    est = NestedComBat(batch=batch).fit(X.loc[train])
    with pytest.raises(ValueError, match="Unseen batch levels"):
        est.transform(X.loc[test])


def test_batch_as_list_of_series():
    """A list of array-likes is accepted as the batch variables."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    as_list = [batch[c] for c in batch.columns]
    from_list = NestedComBat(batch=as_list).fit_transform(X)
    from_df = NestedComBat(batch=batch).fit_transform(X)
    np.testing.assert_allclose(from_list.values, from_df.values, rtol=1e-10, atol=1e-10)


def test_reference_batch_dict_runs():
    """A per-variable reference_batch dict is accepted and produces finite output."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    out = NestedComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        reference_batch={"site": "S1", "scanner": "GE"},
    ).fit_transform(X)
    assert np.isfinite(out.values).all()


def test_reference_batch_str_single_variable():
    """A bare-string reference_batch is allowed with a single batch variable."""
    X, batch, _, _ = simulate_nested_data(n_vars=1, random_state=0)
    out = NestedComBat(batch=batch, reference_batch="S1").fit_transform(X)
    assert np.isfinite(out.values).all()


def test_reference_batch_str_multi_variable_raises():
    """A bare-string reference_batch is ambiguous with several batch variables."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    with pytest.raises(ValueError, match="single level but there are"):
        NestedComBat(batch=batch, reference_batch="S1").fit(X)


def test_reference_batch_unknown_key_raises():
    """reference_batch keys must name actual batch variables."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    with pytest.raises(ValueError, match="not batch"):
        NestedComBat(batch=batch, reference_batch={"nope": "S1"}).fit(X)


def test_unsupported_method_raises():
    """johnson/longitudinal cannot be used as the nested engine."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    for method in ["johnson", "longitudinal"]:
        with pytest.raises(ValueError, match="not supported by NestedComBat"):
            NestedComBat(batch=batch, method=method).fit(X)


def test_invalid_gmm_and_order_metric_raise():
    """Unknown gmm role or order metric is rejected."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    with pytest.raises(ValueError, match="gmm="):
        NestedComBat(batch=batch, gmm="nope").fit(X)
    with pytest.raises(ValueError, match="order_metric="):
        NestedComBat(batch=batch, order_metric="ks").fit(X)


def test_duplicate_batch_names_raise():
    """Duplicated batch variable names are rejected."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    dup = pd.concat([batch["site"].rename("site"), batch["site"].rename("site")], axis=1)
    with pytest.raises(ValueError, match="must be unique"):
        NestedComBat(batch=dup).fit(X)


def test_clone_get_set_params():
    """The estimator clones and exposes its constructor params."""
    _, batch, _, _ = simulate_nested_data(random_state=0)
    est = NestedComBat(batch=batch, method="fortin", optimize_order=True)
    cloned = clone(est)
    cloned.set_params(method="chen", optimize_order=False)
    params = cloned.get_params()
    assert params["method"] == "chen"
    assert params["optimize_order"] is False
    assert NestedComBat(batch=batch).get_params()["max_exhaustive_vars"] == 4
    assert NestedComBat(batch=batch).get_params()["random_state"] is None


def test_pipeline_and_clone_reproducible():
    """Runs inside a Pipeline and reproduces after cloning."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    pipe = Pipeline(
        [
            (
                "combat",
                NestedComBat(batch=batch, discrete_covariates=disc, continuous_covariates=cont),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    out1 = pipe.fit_transform(X)
    out2 = clone(pipe).fit_transform(X)
    np.testing.assert_allclose(out1, out2, rtol=1e-8, atol=1e-8)


def test_get_feature_names_out():
    """Feature names round-trip through the fitted estimator."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    est = NestedComBat(batch=batch).fit(X)
    assert list(est.get_feature_names_out()) == list(X.columns)


def test_nested_matches_manual_sequential_combat():
    """Nested harmonization is exactly single-batch ComBat applied in sequence."""
    X, batch, disc, cont = simulate_nested_data(n_vars=2, random_state=0)
    order = list(batch.columns)
    nested = NestedComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method="fortin",
        optimize_order=False,
    ).fit_transform(X)

    x_current = X
    for name in order:
        model = ComBatModel(method="fortin").fit(
            x_current, batch=batch[name], discrete_covariates=disc, continuous_covariates=cont
        )
        x_current = model.transform(
            x_current, batch=batch[name], discrete_covariates=disc, continuous_covariates=cont
        )
    np.testing.assert_allclose(nested.values, x_current.values, rtol=1e-10, atol=1e-10)


def test_no_spurious_warnings_on_clean_data():
    """A well-specified nested fit on clean data should not warn."""
    X, batch, disc, cont = simulate_nested_data(random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NestedComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont
        ).fit_transform(X)


def test_gmm_not_found_warns_and_proceeds():
    """When no feature yields a balanced split, the GMM is skipped with a warning."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    with pytest.warns(UserWarning, match="balanced two-component"):
        est = NestedComBat(batch=batch, gmm="batch", gmm_min_cluster_frac=0.6).fit(X)
    assert est._gmm_grouping is None
    assert "GMM" not in est.order_


def test_gmm_batch_name_clash_raises():
    """A batch variable already named 'GMM' clashes with gmm='batch'."""
    X, batch, _, _ = simulate_nested_data(latent_split=True, random_state=0)
    batch = batch.rename(columns={"site": "GMM"})
    with pytest.raises(ValueError, match="already named 'GMM'"):
        NestedComBat(batch=batch, gmm="batch").fit(X)


def test_numpy_array_input():
    """Plain numpy arrays (no index) are accepted for X and the batch variables."""
    X, batch, _, _ = simulate_nested_data(random_state=0)
    X_np = X.to_numpy()
    batch_list = [batch[c].to_numpy() for c in batch.columns]
    est = NestedComBat(batch=batch_list, optimize_order=False).fit(X_np)
    out = est.transform(X_np)
    assert out.shape == X_np.shape
    assert np.isfinite(out.values).all()
    assert list(est.get_feature_names_out()) == [f"x{i}" for i in range(X_np.shape[1])]

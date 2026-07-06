import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import simulate_covariate_data, simulate_data

from combatlearn import ComBat
from combatlearn.core import ComBatModel


def test_transform_without_fit_raises():
    """Test that `transform` raises a `ValueError` if not fitted."""
    X, batch = simulate_data()
    model = ComBatModel()
    with pytest.raises(ValueError, match="not fitted"):
        model.transform(X, batch=batch)


def test_unseen_batch_raises_value_error():
    """Test that unseen batch raises a `ValueError`."""
    X, batch = simulate_data()
    model = ComBatModel().fit(X, batch=batch)
    new_batch = pd.Series(["Z"] * len(batch), index=batch.index)
    with pytest.raises(ValueError):
        model.transform(X, batch=new_batch)


def test_single_sample_batch_error():
    """Test that a single sample batch raises a `ValueError`."""
    X, batch = simulate_data()
    batch.iloc[0] = "single"
    with pytest.raises(ValueError):
        ComBatModel().fit(X, batch=batch)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_dtypes_preserved(method):
    """All output columns must remain floating dtypes after correction."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    X_corr = ComBat(batch=batch, method=method, **extra).fit_transform(X)
    assert all(np.issubdtype(dt, np.floating) for dt in X_corr.dtypes)


def test_wrapper_clone_and_pipeline():
    """Test `ComBat` wrapper can be cloned and used in a `Pipeline`."""
    X, batch = simulate_data()
    wrapper = ComBat(batch=batch, parametric=True)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("combat", wrapper),
        ]
    )
    X_corr = pipe.fit_transform(X)
    pipe_clone: Pipeline = clone(pipe)
    X_corr2 = pipe_clone.fit_transform(X)
    np.testing.assert_allclose(X_corr, X_corr2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_no_nan_or_inf_in_output(method):
    """`ComBat` must not introduce NaN or Inf values, for any backend."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    X_corr = ComBat(batch=batch, method=method, **extra).fit_transform(X)
    assert not np.isnan(X_corr.values).any()
    assert not np.isinf(X_corr.values).any()


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_shape_preserved(method):
    """The (n_samples, n_features) shape must be identical pre- and post-ComBat."""
    if method == "johnson":
        X, batch = simulate_data()
        combat = ComBat(batch=batch, method=method).fit(X)
    elif method in ["fortin", "chen"]:
        X, batch, disc, cont = simulate_covariate_data()
        combat = ComBat(
            batch=batch,
            discrete_covariates=disc,
            continuous_covariates=cont,
            method=method,
        ).fit(X)

    X_corr = combat.transform(X)
    assert X_corr.shape == X.shape


def test_johnson_print_warning():
    """Test that a warning is printed when using the Johnson method."""
    X, batch, disc, cont = simulate_covariate_data()
    with pytest.warns(Warning, match="Covariates are ignored when using method='johnson'."):
        _ = ComBat(
            batch=batch,
            discrete_covariates=disc,
            continuous_covariates=cont,
            method="johnson",
        ).fit(X)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_reference_batch_samples_unchanged(method):
    """Samples belonging to the reference batch must come out numerically identical."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    elif method in ["fortin", "chen"]:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    ref_batch = batch.iloc[0]
    combat = ComBat(batch=batch, method=method, reference_batch=ref_batch, **extra).fit(X)
    X_corr = combat.transform(X)

    mask = batch == ref_batch
    np.testing.assert_allclose(X_corr.loc[mask].values, X.loc[mask].values, rtol=0, atol=1e-10)


def test_reference_batch_missing_raises():
    """Asking for a reference batch that doesn't exist should fail."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="not found"):
        ComBat(batch=batch, reference_batch="DOES_NOT_EXIST").fit(X)


@pytest.mark.parametrize("parametric", [True, False])
@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_parametric_vs_nonparametric(parametric, method):
    """Test both parametric and non-parametric modes work without errors."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    combat = ComBat(batch=batch, method=method, parametric=parametric, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


@pytest.mark.parametrize("mean_only", [True, False])
@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_mean_only_mode(mean_only, method):
    """Test mean_only mode works for all methods."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    combat = ComBat(batch=batch, method=method, mean_only=mean_only, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_covbat_cov_thresh_as_float():
    """Test CovBat with covbat_cov_thresh as float."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method="chen",
        covbat_cov_thresh=0.95,
    )
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_covbat_cov_thresh_as_int():
    """Test CovBat with covbat_cov_thresh as int."""
    X, batch, disc, cont = simulate_covariate_data()
    n_components = 10
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method="chen",
        covbat_cov_thresh=n_components,
    )
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()
    assert combat._model._covbat_n_pc == n_components


def test_covbat_cov_thresh_invalid_float_raises():
    """Test that invalid float values for covbat_cov_thresh raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        ComBatModel(covbat_cov_thresh=1.5)
    with pytest.raises(ValueError, match="out of range"):
        ComBatModel(covbat_cov_thresh=0.0)


def test_covbat_cov_thresh_invalid_int_raises():
    """Test that invalid int values for covbat_cov_thresh raise ValueError."""
    with pytest.raises(ValueError, match="invalid"):
        ComBatModel(covbat_cov_thresh=0)


def test_covbat_cov_thresh_invalid_type_raises():
    """Test that invalid types for covbat_cov_thresh raise TypeError."""
    with pytest.raises(TypeError, match="must be float or int"):
        ComBatModel(covbat_cov_thresh="invalid")


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_index_preserved(method):
    """Test that the index is preserved after transformation."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    custom_index = pd.Index([f"sample_{i}" for i in range(len(X))])
    X.index = custom_index
    batch.index = custom_index
    if method != "johnson":
        disc.index = custom_index
        cont.index = custom_index

    combat = ComBat(batch=batch, method=method, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.index.equals(custom_index)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_column_names_preserved(method):
    """Test that column names are preserved after transformation."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    custom_columns = [f"feature_{i}" for i in range(X.shape[1])]
    X.columns = custom_columns

    combat = ComBat(batch=batch, method=method, **extra)
    X_corr = combat.fit_transform(X)
    assert list(X_corr.columns) == custom_columns


def test_invalid_method_raises():
    """Test that an invalid method raises ValueError."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="not recognized"):
        ComBatModel(method="invalid").fit(X, batch=batch)


def test_fit_nan_in_X_raises():
    """Fitting with NaN in X must raise ValueError."""
    X, batch = simulate_data()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ComBat(batch=batch).fit(X)


def test_fit_inf_in_X_raises():
    """Fitting with Inf in X must raise ValueError."""
    X, batch = simulate_data()
    X.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        ComBat(batch=batch).fit(X)


def test_transform_nan_in_X_raises():
    """Transforming with NaN in X must raise ValueError."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch).fit(X)
    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        combat.transform(X_bad)


def test_fit_nan_in_batch_raises():
    """Fitting with NaN in batch must raise ValueError."""
    X, batch = simulate_data()
    batch.iloc[0] = np.nan
    with pytest.raises(ValueError, match=r"batch contains.*NaN"):
        ComBat(batch=batch).fit(X)


def test_fit_nan_in_discrete_covariates_raises():
    """Fitting with NaN in discrete covariates must raise ValueError."""
    X, batch, disc, cont = simulate_covariate_data()
    disc.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"discrete_covariates contains.*NaN"):
        ComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
        ).fit(X)


def test_fit_nan_in_continuous_covariates_raises():
    """Fitting with NaN in continuous covariates must raise ValueError."""
    X, batch, disc, cont = simulate_covariate_data()
    cont.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
        ).fit(X)


def test_get_feature_names_out_dataframe():
    """get_feature_names_out returns column names from DataFrame input."""
    X, batch = simulate_data(n_samples=100, n_features=10)
    X.columns = [f"feat_{i}" for i in range(10)]
    combat = ComBat(batch=batch).fit(X)
    names = combat.get_feature_names_out()
    assert list(names) == [f"feat_{i}" for i in range(10)]


def test_get_feature_names_out_ndarray():
    """get_feature_names_out returns default names for numpy input."""
    X, batch = simulate_data(n_samples=100, n_features=10)
    combat = ComBat(batch=batch).fit(X.values)
    names = combat.get_feature_names_out()
    assert list(names) == [f"x{i}" for i in range(10)]


def test_get_feature_names_out_not_fitted_raises():
    """get_feature_names_out before fit raises NotFittedError."""
    _X, batch = simulate_data()
    combat = ComBat(batch=batch)
    with pytest.raises(NotFittedError):
        combat.get_feature_names_out()


def test_set_output_pandas_in_pipeline():
    """Pipeline with set_output(transform='pandas') works with ComBat."""
    X, batch = simulate_data(n_samples=100, n_features=10)
    X.columns = [f"feat_{i}" for i in range(10)]
    pipe = Pipeline([("combat", ComBat(batch=batch))])
    pipe.set_output(transform="pandas")
    result = pipe.fit_transform(X)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [f"feat_{i}" for i in range(10)]


def test_set_params_updates_method():
    """set_params must propagate to the model created at fit time."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch, method="johnson")
    combat.set_params(method="fortin")
    combat.fit(X)
    assert combat._model.method == "fortin"


@pytest.mark.parametrize("method", ["fortin", "chen"])
def test_transform_subset_of_batches(method):
    """Transform with a subset of fitted batches must not crash."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method=method,
    ).fit(X)

    mask = batch != "C"
    X_sub = X.loc[mask]
    X_corr = combat.transform(X_sub)
    assert X_corr.shape == X_sub.shape
    assert not np.isnan(X_corr.values).any()


@pytest.mark.parametrize("method", ["fortin", "chen"])
def test_transform_new_covariate_levels(method):
    """Transform with unseen discrete covariate levels must not crash."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method=method,
    ).fit(X)

    disc_new = disc.copy()
    disc_new.iloc[0, 0] = "diag_3"
    combat_new = ComBat(
        batch=batch,
        discrete_covariates=disc_new,
        continuous_covariates=cont,
        method=method,
    )
    combat_new._model = combat._model
    combat_new.feature_names_in_ = combat.feature_names_in_
    X_corr = combat_new.transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_error_message_includes_received_value():
    """Error messages should include the received value."""
    with pytest.raises(ValueError, match=r"1\.5"):
        ComBatModel(covbat_cov_thresh=1.5)
    with pytest.raises(TypeError, match="str"):
        ComBatModel(covbat_cov_thresh="bad")


def test_error_message_batch_too_small_includes_count():
    """Batch-too-small error should include sample count and suggestion."""
    X, batch = simulate_data()
    batch.iloc[0] = "single"
    with pytest.raises(ValueError, match=r"1 sample.*Consider merging"):
        ComBatModel().fit(X, batch=batch)


def test_error_message_reference_batch_lists_available():
    """reference_batch error should list available batches."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="Available batches"):
        ComBatModel(reference_batch="MISSING").fit(X, batch=batch)


def test_near_zero_variance_warns():
    """Near-zero variance features should trigger a warning."""
    X, batch = simulate_data(n_samples=60, n_features=10)
    X["constant"] = 0.0
    with pytest.warns(UserWarning, match="near-zero variance"):
        ComBat(batch=batch).fit(X)


def test_imbalanced_batches_warns():
    """Highly imbalanced batches should trigger a warning."""
    rng = np.random.default_rng(42)
    n_large, n_small = 200, 5
    n = n_large + 2 * n_small
    X = pd.DataFrame(rng.standard_normal((n, 10)))
    batch = pd.Series(["A"] * n_large + ["B"] * n_small + ["C"] * n_small)
    with pytest.warns(UserWarning, match="imbalanced"):
        ComBat(batch=batch).fit(X)


def test_collinear_covariate_warns():
    """Covariate perfectly collinear with batch should trigger a warning."""
    X, batch, _disc, _cont = simulate_covariate_data()
    disc_collinear = batch.to_frame("cov")
    with pytest.warns(UserWarning, match="rank-deficient"):
        ComBat(
            batch=batch,
            discrete_covariates=disc_collinear,
            method="fortin",
        ).fit(X)


def test_balanced_batches_no_imbalance_warning():
    """Balanced batches should not trigger imbalance warning."""
    X, batch = simulate_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ComBat(batch=batch).fit(X)


@pytest.mark.parametrize(
    "alias,canonical",
    [("classic_combat", "johnson"), ("neurocombat", "fortin"), ("covbat", "chen")],
)
def test_method_alias_matches_canonical(alias, canonical):
    """A literature alias must produce the same result as its canonical author name."""
    if canonical == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}
    out_alias = ComBat(batch=batch, method=alias, **extra).fit_transform(X)
    out_canon = ComBat(batch=batch, method=canonical, **extra).fit_transform(X)
    np.testing.assert_allclose(out_alias.values, out_canon.values, rtol=1e-10, atol=1e-10)


def test_method_alias_case_and_separator_insensitive():
    """Alias matching ignores case, hyphens, underscores, and spaces."""
    X, batch, disc, cont = simulate_covariate_data()
    base = ComBat(
        batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
    ).fit_transform(X)
    for name in ["neuroCombat", "neuro-combat", "NEUROCOMBAT", "neuro_combat"]:
        out = ComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont, method=name
        ).fit_transform(X)
        np.testing.assert_allclose(out.values, base.values, rtol=1e-10, atol=1e-10)


def test_classic_combat_separator_variants():
    """'classic_combat', 'classic-combat', 'ClassicComBat' all resolve to johnson."""
    X, batch = simulate_data()
    base = ComBat(batch=batch, method="johnson").fit_transform(X)
    for name in ["classic_combat", "classic-combat", "ClassicComBat"]:
        out = ComBat(batch=batch, method=name).fit_transform(X)
        np.testing.assert_allclose(out.values, base.values, rtol=1e-10, atol=1e-10)


def test_invalid_method_message_lists_aliases():
    """The unrecognized-method error mentions the alias names."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="classic_combat"):
        ComBat(batch=batch, method="not_a_method").fit(X)

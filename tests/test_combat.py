import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from combatlearn import ComBatModel, ComBat
from utils import simulate_data, simulate_covariate_data


def test_transform_without_fit_raises():
    """
    Test that `transform` raises a `NotFittedError` if not fitted.
    """
    X, batch = simulate_data()
    model = ComBatModel()
    with pytest.raises(NotFittedError):
        model.transform(X, batch=batch)


def test_unseen_batch_raises_value_error():
    """
    Test that unseen batch raises a `ValueError`.
    """
    X, batch = simulate_data()
    model = ComBatModel().fit(X, batch=batch)
    new_batch = pd.Series(["Z"] * len(batch), index=batch.index)
    with pytest.raises(ValueError):
        model.transform(X, batch=new_batch)


def test_single_sample_batch_error():
    """
    Test that a single sample batch raises a `ValueError`.
    """
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
    else:                                  # fortin  or  chen
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

    X_corr = ComBat(batch=batch, method=method, **extra).fit_transform(X)
    assert all(np.issubdtype(dt, np.floating) for dt in X_corr.dtypes)

def test_wrapper_clone_and_pipeline():
    """
    Test `ComBat` wrapper can be cloned and used in a `Pipeline`.
    """
    X, batch = simulate_data()
    wrapper = ComBat(batch=batch, parametric=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("combat", wrapper),
    ])
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
    else:                                  # fortin  or  chen
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

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
    """
    Test that a warning is printed when using the Johnson method.
    """
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
    """
    Samples belonging to the reference batch must come out *numerically identical*
    (within floating-point jitter) after correction.
    """
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    elif method in ["fortin", "chen"]:
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

    ref_batch = batch.iloc[0]
    combat = ComBat(batch=batch, method=method,
                    reference_batch=ref_batch, **extra).fit(X)
    X_corr = combat.transform(X)

    mask = batch == ref_batch
    np.testing.assert_allclose(X_corr.loc[mask].values,
                               X.loc[mask].values,
                               rtol=0, atol=1e-10)


def test_reference_batch_missing_raises():
    """
    Asking for a reference batch that doesn't exist should fail.
    """
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="not present"):
        ComBat(batch=batch, reference_batch="DOES_NOT_EXIST").fit(X)
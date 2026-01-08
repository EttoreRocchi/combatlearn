import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from combatlearn import ComBat
from combatlearn.combat import ComBatModel
from utils import simulate_data, simulate_covariate_data


def test_transform_without_fit_raises():
    """
    Test that `transform` raises a `ValueError` if not fitted.
    """
    X, batch = simulate_data()
    model = ComBatModel()
    with pytest.raises(ValueError, match="not fitted"):
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


@pytest.mark.parametrize("parametric", [True, False])
@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_parametric_vs_nonparametric(parametric, method):
    """
    Test both parametric and non-parametric modes work without errors.
    """
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

    combat = ComBat(batch=batch, method=method, parametric=parametric, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


@pytest.mark.parametrize("mean_only", [True, False])
@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_mean_only_mode(mean_only, method):
    """
    Test mean_only mode works for all methods.
    """
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

    combat = ComBat(batch=batch, method=method, mean_only=mean_only, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_covbat_cov_thresh_as_float():
    """
    Test CovBat with covbat_cov_thresh as float (cumulative variance threshold).
    """
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
    """
    Test CovBat with covbat_cov_thresh as int (number of components).
    """
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
    """
    Test that invalid float values for covbat_cov_thresh raise ValueError.
    """
    with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
        ComBatModel(covbat_cov_thresh=1.5)

    with pytest.raises(ValueError, match="must be in \\(0, 1\\]"):
        ComBatModel(covbat_cov_thresh=0.0)


def test_covbat_cov_thresh_invalid_int_raises():
    """
    Test that invalid int values for covbat_cov_thresh raise ValueError.
    """
    with pytest.raises(ValueError, match="must be >= 1"):
        ComBatModel(covbat_cov_thresh=0)


def test_covbat_cov_thresh_invalid_type_raises():
    """
    Test that invalid types for covbat_cov_thresh raise TypeError.
    """
    with pytest.raises(TypeError, match="must be float or int"):
        ComBatModel(covbat_cov_thresh="invalid")


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_index_preserved(method):
    """
    Test that the index is preserved after transformation.
    """
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

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
    """
    Test that column names are preserved after transformation.
    """
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = dict(discrete_covariates=disc, continuous_covariates=cont)

    custom_columns = [f"feature_{i}" for i in range(X.shape[1])]
    X.columns = custom_columns

    combat = ComBat(batch=batch, method=method, **extra)
    X_corr = combat.fit_transform(X)
    assert list(X_corr.columns) == custom_columns


def test_plot_transformation_static_2d():
    """
    Test plot_transformation with static 2D PCA visualization.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = combat.plot_transformation(
        X,
        reduction_method='pca',
        n_components=2,
        plot_type='static'
    )
    assert fig is not None


def test_plot_transformation_static_3d():
    """
    Test plot_transformation with static 3D PCA visualization.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = combat.plot_transformation(
        X,
        reduction_method='pca',
        n_components=3,
        plot_type='static'
    )
    assert fig is not None


def test_plot_transformation_return_embeddings():
    """
    Test that plot_transformation can return embeddings.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig, embeddings = combat.plot_transformation(
        X,
        reduction_method='pca',
        n_components=2,
        plot_type='static',
        return_embeddings=True
    )
    assert fig is not None
    assert 'original' in embeddings
    assert 'transformed' in embeddings
    assert embeddings['original'].shape == (100, 2)
    assert embeddings['transformed'].shape == (100, 2)


def test_plot_transformation_invalid_method_raises():
    """
    Test that invalid reduction_method raises ValueError.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="reduction_method must be"):
        combat.plot_transformation(X, reduction_method='invalid')


def test_plot_transformation_invalid_n_components_raises():
    """
    Test that invalid n_components raises ValueError.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="n_components must be 2 or 3"):
        combat.plot_transformation(X, n_components=4)


def test_invalid_method_raises():
    """
    Test that an invalid method raises ValueError.
    """
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="method must be"):
        ComBatModel(method="invalid").fit(X, batch=batch)


def test_compute_metrics_caches_in_metrics_property():
    """
    Test that compute_metrics=True caches metrics in metrics_ property.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)

    combat = ComBat(batch=batch, method="johnson", compute_metrics=True)
    X_corr = combat.fit_transform(X)

    assert combat.metrics_ is not None
    assert 'batch_effect' in combat.metrics_
    assert 'preservation' in combat.metrics_
    assert 'alignment' in combat.metrics_


def test_compute_metrics_false_returns_none():
    """
    Test that metrics_ is None when compute_metrics=False.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)

    combat = ComBat(batch=batch, method="johnson", compute_metrics=False)
    X_corr = combat.fit_transform(X)

    assert combat.metrics_ is None


def test_compute_batch_metrics_returns_correct_structure():
    """
    Test that compute_batch_metrics returns the expected structure.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)

    combat = ComBat(batch=batch, method="johnson")
    combat.fit(X)

    metrics = combat.compute_batch_metrics(X, k_neighbors=[5, 10])

    assert 'batch_effect' in metrics
    assert 'silhouette' in metrics['batch_effect']
    assert 'davies_bouldin' in metrics['batch_effect']
    assert 'kbet' in metrics['batch_effect']
    assert 'lisi' in metrics['batch_effect']
    assert 'variance_ratio' in metrics['batch_effect']

    for metric_name in ['silhouette', 'davies_bouldin', 'kbet', 'lisi', 'variance_ratio']:
        metric_vals = metrics['batch_effect'][metric_name]
        assert 'before' in metric_vals
        assert 'after' in metric_vals

    assert 'preservation' in metrics
    assert 'knn' in metrics['preservation']
    assert 5 in metrics['preservation']['knn']
    assert 10 in metrics['preservation']['knn']
    assert 'distance_correlation' in metrics['preservation']

    assert 'alignment' in metrics
    assert 'centroid_distance' in metrics['alignment']
    assert 'levene_statistic' in metrics['alignment']


def test_compute_batch_metrics_not_fitted_raises():
    """
    Test that compute_batch_metrics raises ValueError if not fitted.
    """
    X, batch = simulate_data(n_samples=100, n_features=20)

    combat = ComBat(batch=batch, method="johnson")

    with pytest.raises(ValueError, match="not fitted"):
        combat.compute_batch_metrics(X)


def test_compute_batch_metrics_pca_components_validation():
    """
    Test that pca_components must be less than min(n_samples, n_features).
    """
    X, batch = simulate_data(n_samples=100, n_features=20)

    combat = ComBat(batch=batch, method="johnson")
    combat.fit(X)

    # pca_components >= n_features should raise
    with pytest.raises(ValueError, match="pca_components.*must be less than"):
        combat.compute_batch_metrics(X, pca_components=20)

    # pca_components > n_features should raise
    with pytest.raises(ValueError, match="pca_components.*must be less than"):
        combat.compute_batch_metrics(X, pca_components=50)

    # Valid pca_components should work
    metrics = combat.compute_batch_metrics(X, pca_components=10)
    assert metrics is not None


def test_compute_batch_metrics_no_pca_default():
    """
    Test that metrics are computed in original feature space by default (no PCA).
    """
    X, batch = simulate_data(n_samples=100, n_features=20)

    combat = ComBat(batch=batch, method="johnson")
    combat.fit(X)

    # Default (pca_components=None) should work
    metrics = combat.compute_batch_metrics(X)
    assert metrics is not None
    assert 'batch_effect' in metrics
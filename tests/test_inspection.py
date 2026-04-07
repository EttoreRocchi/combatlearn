import numpy as np
import pytest
from utils import simulate_covariate_data, simulate_data

from combatlearn import ComBat
from combatlearn.inspection import feature_batch_diagnostics, summary


def test_feature_diagnostics_shape():
    """Test that feature_batch_diagnostics returns correct shape."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat)
    assert importance.shape == (25, 3)
    assert list(importance.columns) == ["location", "scale", "combined"]


def test_feature_diagnostics_location_rms_weighted():
    """Test that location is weighted RMS of gamma across batches."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat)
    gamma_star = combat._model._gamma_star
    n_per_batch = combat._model._n_per_batch
    weights = np.array(
        [n_per_batch[str(lvl)] for lvl in combat._model._batch_levels],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    expected_location = np.sqrt((weights[:, np.newaxis] * gamma_star**2).sum(axis=0))

    np.testing.assert_allclose(
        importance.sort_index()["location"].values,
        expected_location,
        rtol=1e-10,
    )


def test_feature_diagnostics_scale_rms_weighted():
    """Test that scale is weighted RMS of log(delta) across batches."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat)
    delta_star = combat._model._delta_star
    n_per_batch = combat._model._n_per_batch
    weights = np.array(
        [n_per_batch[str(lvl)] for lvl in combat._model._batch_levels],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    expected_scale = np.sqrt((weights[:, np.newaxis] * np.log(delta_star) ** 2).sum(axis=0))

    np.testing.assert_allclose(
        importance.sort_index()["scale"].values,
        expected_scale,
        rtol=1e-10,
    )


def test_feature_diagnostics_location_rms_unweighted():
    """Test that location is unweighted RMS of gamma when weighted=False."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat, weighted=False)
    gamma_star = combat._model._gamma_star
    expected_location = np.sqrt((gamma_star**2).mean(axis=0))

    np.testing.assert_allclose(
        importance.sort_index()["location"].values,
        expected_location,
        rtol=1e-10,
    )


def test_feature_diagnostics_scale_rms_unweighted():
    """Test that scale is unweighted RMS of log(delta) when weighted=False."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat, weighted=False)
    delta_star = combat._model._delta_star
    expected_scale = np.sqrt((np.log(delta_star) ** 2).mean(axis=0))

    np.testing.assert_allclose(
        importance.sort_index()["scale"].values,
        expected_scale,
        rtol=1e-10,
    )


def test_feature_diagnostics_combined_euclidean():
    """Test that combined is Euclidean norm of location and scale."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat)
    expected_combined = np.sqrt(importance["location"] ** 2 + importance["scale"] ** 2)

    np.testing.assert_allclose(
        importance["combined"].values,
        expected_combined.values,
        rtol=1e-10,
    )


def test_feature_diagnostics_mean_only_zero_scale():
    """Test that scale is zero when mean_only=True."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson", mean_only=True).fit(X)

    importance = feature_batch_diagnostics(combat)

    np.testing.assert_array_equal(importance["scale"].values, 0.0)
    np.testing.assert_allclose(
        importance["location"].values,
        importance["combined"].values,
        rtol=1e-10,
    )


def test_feature_diagnostics_mode_magnitude():
    """Test that magnitude mode returns raw values."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat, mode="magnitude")

    # Values should not sum to 1
    assert importance["combined"].sum() != pytest.approx(1.0, rel=0.01)


def test_feature_diagnostics_mode_distribution_sums_to_one():
    """Test that distribution mode normalizes each column to sum to 1."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    importance = feature_batch_diagnostics(combat, mode="distribution")

    assert importance["location"].sum() == pytest.approx(1.0, rel=1e-10)
    assert importance["scale"].sum() == pytest.approx(1.0, rel=1e-10)
    assert importance["combined"].sum() == pytest.approx(1.0, rel=1e-10)


def test_feature_diagnostics_invalid_mode_raises():
    """Test that invalid mode raises ValueError."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="mode must be"):
        feature_batch_diagnostics(combat, mode="invalid")


def test_feature_diagnostics_not_fitted_raises():
    """Test that feature_batch_diagnostics raises ValueError if not fitted."""
    _X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson")

    with pytest.raises(ValueError, match="not fitted"):
        feature_batch_diagnostics(combat)


def test_summary_contains_method():
    """summary() output contains the method name."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch, method="johnson").fit(X)
    s = summary(combat)
    assert "johnson" in s


def test_summary_contains_batch_info():
    """summary() output contains batch count and sample counts."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch).fit(X)
    s = summary(combat)
    assert "Number of batches: 3" in s
    assert "A:" in s
    assert "B:" in s
    assert "C:" in s


def test_summary_not_fitted_raises():
    """summary() before fit raises ValueError."""
    _X, batch = simulate_data()
    combat = ComBat(batch=batch)
    with pytest.raises(ValueError, match="not fitted"):
        summary(combat)


def test_summary_variance_explained():
    """summary() should contain variance explained before and after."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch).fit(X)
    combat.transform(X)
    s = summary(combat)
    assert "Batch var. explained (before)" in s
    assert "Batch var. explained (after)" in s


def test_summary_variance_explained_before_only():
    """summary() after fit() only should show before but not after."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch).fit(X)
    s = summary(combat)
    assert "Batch var. explained (before)" in s
    assert "Batch var. explained (after)" not in s


def test_summary_convergence_info():
    """summary() should contain convergence info."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch).fit(X)
    s = summary(combat)
    assert "converged" in s


def test_summary_condition_number_fortin():
    """summary() for fortin method should show condition number."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
    ).fit(X)
    s = summary(combat)
    assert "Design matrix condition number" in s


def test_summary_condition_number_absent_johnson():
    """summary() for johnson method should NOT show condition number."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch, method="johnson").fit(X)
    s = summary(combat)
    assert "Design matrix condition number" not in s

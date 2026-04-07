import pytest
from utils import simulate_data

from combatlearn import ComBat
from combatlearn.metrics import compute_batch_metrics


def test_compute_batch_metrics_returns_correct_structure():
    """Test that compute_batch_metrics returns the expected structure."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    metrics = compute_batch_metrics(combat, X, k_neighbors=[5, 10])

    assert "batch_effect" in metrics
    assert "silhouette" in metrics["batch_effect"]
    assert "davies_bouldin" in metrics["batch_effect"]
    assert "kbet" in metrics["batch_effect"]
    assert "lisi" in metrics["batch_effect"]
    assert "variance_ratio" in metrics["batch_effect"]

    for metric_name in [
        "silhouette",
        "davies_bouldin",
        "kbet",
        "lisi",
        "variance_ratio",
    ]:
        metric_vals = metrics["batch_effect"][metric_name]
        assert "before" in metric_vals
        assert "after" in metric_vals

    assert "preservation" in metrics
    assert "knn" in metrics["preservation"]
    assert 5 in metrics["preservation"]["knn"]
    assert 10 in metrics["preservation"]["knn"]
    assert "distance_correlation" in metrics["preservation"]

    assert "alignment" in metrics
    assert "centroid_distance" in metrics["alignment"]
    assert "levene_statistic" in metrics["alignment"]


def test_compute_batch_metrics_not_fitted_raises():
    """Test that compute_batch_metrics raises ValueError if not fitted."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson")

    with pytest.raises(ValueError, match="not fitted"):
        compute_batch_metrics(combat, X)


def test_compute_batch_metrics_pca_components_validation():
    """Test that pca_components must be less than min(n_samples, n_features)."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match=r"pca_components.*must be less than"):
        compute_batch_metrics(combat, X, pca_components=20)

    with pytest.raises(ValueError, match=r"pca_components.*must be less than"):
        compute_batch_metrics(combat, X, pca_components=50)

    metrics = compute_batch_metrics(combat, X, pca_components=10)
    assert metrics is not None


def test_compute_batch_metrics_no_pca_default():
    """Test that metrics are computed in original feature space by default (no PCA)."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    metrics = compute_batch_metrics(combat, X)
    assert metrics is not None
    assert "batch_effect" in metrics


def test_metrics_nn_algorithm_ball_tree():
    """compute_batch_metrics with nn_algorithm='ball_tree' must work."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch).fit(X)
    metrics = compute_batch_metrics(combat, X, k_neighbors=[5], nn_algorithm="ball_tree")
    assert "batch_effect" in metrics


def test_metrics_nn_algorithm_kd_tree():
    """compute_batch_metrics with nn_algorithm='kd_tree' must work."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch).fit(X)
    metrics = compute_batch_metrics(combat, X, k_neighbors=[5], nn_algorithm="kd_tree")
    assert "batch_effect" in metrics


def test_metrics_nn_algorithm_brute():
    """compute_batch_metrics with nn_algorithm='brute' must work."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch).fit(X)
    metrics = compute_batch_metrics(combat, X, k_neighbors=[5], nn_algorithm="brute")
    assert "batch_effect" in metrics


def test_metrics_nn_algorithm_invalid_raises():
    """compute_batch_metrics with invalid nn_algorithm must raise ValueError."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch).fit(X)
    with pytest.raises(ValueError, match="nn_algorithm"):
        compute_batch_metrics(combat, X, nn_algorithm="invalid")

import pytest
from utils import simulate_data

from combatlearn import ComBat
from combatlearn.visualization import (
    plot_batch_effect_heatmap,
    plot_feature_diagnostics,
    plot_transformation,
)


def test_plot_transformation_static_2d():
    """Test plot_transformation with static 2D PCA visualization."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_transformation(combat, X, reduction_method="pca", n_components=2, plot_type="static")
    assert fig is not None


def test_plot_transformation_static_3d():
    """Test plot_transformation with static 3D PCA visualization."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_transformation(combat, X, reduction_method="pca", n_components=3, plot_type="static")
    assert fig is not None


def test_plot_transformation_return_embeddings():
    """Test that plot_transformation can return embeddings."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig, embeddings = plot_transformation(
        combat,
        X,
        reduction_method="pca",
        n_components=2,
        plot_type="static",
        return_embeddings=True,
    )
    assert fig is not None
    assert "original" in embeddings
    assert "transformed" in embeddings
    assert embeddings["original"].shape == (100, 2)
    assert embeddings["transformed"].shape == (100, 2)


def test_plot_transformation_invalid_method_raises():
    """Test that invalid reduction_method raises ValueError."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="reduction_method must be"):
        plot_transformation(combat, X, reduction_method="invalid")


def test_plot_transformation_invalid_n_components_raises():
    """Test that invalid n_components raises ValueError."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="n_components must be 2 or 3"):
        plot_transformation(combat, X, n_components=4)


def test_plot_transformation_not_fitted_raises():
    """Test that plot_transformation raises ValueError if not fitted."""
    X, batch = simulate_data(n_samples=100, n_features=20)
    combat = ComBat(batch=batch, method="johnson")

    with pytest.raises(ValueError, match="not fitted"):
        plot_transformation(combat, X)


def test_plot_feature_diagnostics_kind_location():
    """Test plot_feature_diagnostics with kind='location'."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, kind="location")
    assert fig is not None


def test_plot_feature_diagnostics_kind_scale():
    """Test plot_feature_diagnostics with kind='scale'."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, kind="scale")
    assert fig is not None


def test_plot_feature_diagnostics_kind_combined_grouped():
    """Test plot_feature_diagnostics with kind='combined' shows grouped bars."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, kind="combined")
    assert fig is not None

    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None


def test_plot_feature_diagnostics_combined_diverging():
    """Test kind='combined' with layout='diverging' renders two side panels."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, kind="combined", layout="diverging")
    assert fig is not None

    # Two panels: Location (x-axis inverted -> grows leftward) and Scale.
    assert len(fig.axes) == 2
    ax_loc, ax_scale = fig.axes
    assert ax_loc.get_title() == "Location"
    assert ax_scale.get_title() == "Scale"
    lo, hi = ax_loc.get_xlim()
    assert lo > hi  # inverted axis


def test_plot_feature_diagnostics_diverging_requires_combined():
    """layout='diverging' is only valid for kind='combined'."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="only available for kind='combined'"):
        plot_feature_diagnostics(combat, kind="location", layout="diverging")


def test_plot_feature_diagnostics_invalid_layout_raises():
    """An unknown layout value raises ValueError."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    with pytest.raises(ValueError, match="layout must be"):
        plot_feature_diagnostics(combat, layout="bogus")


def test_plot_feature_diagnostics_mode_magnitude():
    """Test plot_feature_diagnostics with mode='magnitude'."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, mode="magnitude")
    assert fig is not None


def test_plot_feature_diagnostics_mode_distribution():
    """Test plot_feature_diagnostics with mode='distribution'."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, mode="distribution")
    assert fig is not None


def test_plot_feature_diagnostics_distribution_shows_cumulative(capsys):
    """Test that distribution mode prints cumulative contribution."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson").fit(X)

    fig = plot_feature_diagnostics(combat, mode="distribution", top_n=10)
    assert fig is not None

    captured = capsys.readouterr()
    assert "features explain" in captured.out
    assert "%" in captured.out


def test_plot_feature_diagnostics_not_fitted_raises():
    """Test that plot_feature_diagnostics raises ValueError if not fitted."""
    _X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, method="johnson")

    with pytest.raises(ValueError, match="not fitted"):
        plot_feature_diagnostics(combat)


def test_heatmap_returns_figure():
    """plot_batch_effect_heatmap returns a matplotlib Figure."""
    import matplotlib.figure

    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch).fit(X)
    fig = plot_batch_effect_heatmap(combat, top_n=10)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_heatmap_top_n():
    """plot_batch_effect_heatmap respects top_n parameter."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch).fit(X)
    fig = plot_batch_effect_heatmap(combat, top_n=10)
    # With mean_only=False, 2 subplots
    assert len(fig.axes) >= 2


def test_heatmap_mean_only():
    """plot_batch_effect_heatmap with mean_only=True shows only 1 heatmap."""
    X, batch = simulate_data(n_samples=100, n_features=25)
    combat = ComBat(batch=batch, mean_only=True).fit(X)
    fig = plot_batch_effect_heatmap(combat, top_n=10)
    assert "gamma" in fig.axes[0].get_title().lower()


def test_heatmap_not_fitted_raises():
    """plot_batch_effect_heatmap before fit raises ValueError."""
    _X, batch = simulate_data()
    combat = ComBat(batch=batch)
    with pytest.raises(ValueError, match="not fitted"):
        plot_batch_effect_heatmap(combat)

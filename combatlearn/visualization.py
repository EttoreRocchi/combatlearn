"""Visualization utilities for ComBat batch correction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._utils import _subset
from .core import ArrayLike, FloatArray
from .inspection import feature_batch_diagnostics

if TYPE_CHECKING:
    from .sklearn_api import ComBat


def _scatter_panel(
    ax: Any,
    X: FloatArray,
    batch_labels: pd.Series,
    unique_batches: pd.Series,
    colors: npt.NDArray[Any],
    n_components: int,
    point_size: int,
    alpha: float,
) -> None:
    """Scatter one panel (before or after) onto the given axes."""
    for i, batch in enumerate(unique_batches):
        mask = batch_labels == batch
        coords = [X[mask, j] for j in range(n_components)]
        ax.scatter(
            *coords,
            c=[colors[i]],
            s=point_size,
            alpha=alpha,
            label=f"Batch {batch}",
            edgecolors="black",
            linewidth=0.5,
        )


def _create_static_plot(
    X_orig: FloatArray,
    X_trans: FloatArray,
    batch_labels: pd.Series,
    method: str,
    n_components: int,
    figsize: tuple[int, int],
    alpha: float,
    point_size: int,
    cmap: str,
    title: str | None,
    show_legend: bool,
) -> Any:
    """Create static plots using matplotlib."""
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)

    unique_batches = batch_labels.drop_duplicates()
    n_batches = len(unique_batches)

    if n_batches <= 10:
        colors = matplotlib.colormaps.get_cmap(cmap)(np.linspace(0, 1, n_batches))
    else:
        colors = matplotlib.colormaps.get_cmap("tab20")(np.linspace(0, 1, n_batches))

    if n_components == 2:
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    else:
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

    scatter_kwargs = {
        "batch_labels": batch_labels,
        "unique_batches": unique_batches,
        "colors": colors,
        "n_components": n_components,
        "point_size": point_size,
        "alpha": alpha,
    }
    _scatter_panel(ax1, X_orig, **scatter_kwargs)
    _scatter_panel(ax2, X_trans, **scatter_kwargs)

    method_upper = method.upper()
    for ax, label in [(ax1, "Before"), (ax2, "After")]:
        ax.set_title(f"{label} ComBat correction\n({method_upper})")
        ax.set_xlabel(f"{method_upper}1")
        ax.set_ylabel(f"{method_upper}2")
        if n_components == 3:
            ax.set_zlabel(f"{method_upper}3")  # type: ignore[attr-defined]

    if show_legend and n_batches <= 20:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    if title is None:
        title = f"ComBat correction effect visualized with {method_upper}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    return fig


def _add_plotly_traces(
    fig: Any,
    X: FloatArray,
    batch_labels: pd.Series,
    unique_batches: pd.Series,
    batch_to_color: dict[Any, Any],
    n_components: int,
    col: int,
    showlegend: bool,
) -> None:
    """Add plotly scatter traces for one panel (before or after)."""
    import plotly.graph_objects as go

    for batch in unique_batches:
        mask = batch_labels == batch
        if n_components == 2:
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    name=f"Batch {batch}",
                    marker={
                        "size": 8,
                        "color": batch_to_color[batch],
                        "line": {"width": 1, "color": "black"},
                    },
                    showlegend=showlegend,
                ),
                row=1,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    z=X[mask, 2],
                    mode="markers",
                    name=f"Batch {batch}",
                    marker={
                        "size": 5,
                        "color": batch_to_color[batch],
                        "line": {"width": 0.5, "color": "black"},
                    },
                    showlegend=showlegend,
                ),
                row=1,
                col=col,
            )


def _create_interactive_plot(
    X_orig: FloatArray,
    X_trans: FloatArray,
    batch_labels: pd.Series,
    method: str,
    n_components: int,
    cmap: str,
    title: str | None,
    show_legend: bool,
) -> Any:
    """Create interactive plots using plotly."""
    import matplotlib
    import matplotlib.colors as mcolors
    from plotly.subplots import make_subplots

    method_upper = method.upper()
    if n_components == 2:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Before ComBat correction ({method_upper})",
                f"After ComBat correction ({method_upper})",
            ),
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            subplot_titles=(
                f"Before ComBat correction ({method_upper})",
                f"After ComBat correction ({method_upper})",
            ),
        )

    unique_batches = batch_labels.drop_duplicates()

    n_batches = len(unique_batches)
    cmap_func = matplotlib.colormaps.get_cmap(cmap)
    color_list = [mcolors.to_hex(cmap_func(i / max(n_batches - 1, 1))) for i in range(n_batches)]
    batch_to_color = dict(zip(unique_batches, color_list, strict=True))

    trace_kwargs = {
        "batch_labels": batch_labels,
        "unique_batches": unique_batches,
        "batch_to_color": batch_to_color,
        "n_components": n_components,
    }
    _add_plotly_traces(fig, X_orig, **trace_kwargs, col=1, showlegend=False)  # type: ignore[arg-type]
    _add_plotly_traces(fig, X_trans, **trace_kwargs, col=2, showlegend=show_legend)  # type: ignore[arg-type]

    if title is None:
        title = f"ComBat correction effect visualized with {method_upper}"

    fig.update_layout(
        title=title,
        title_font_size=16,
        height=600,
        showlegend=show_legend,
        hovermode="closest",
    )

    axis_labels = [f"{method_upper}{i + 1}" for i in range(n_components)]

    if n_components == 2:
        fig.update_xaxes(title_text=axis_labels[0])
        fig.update_yaxes(title_text=axis_labels[1])
    else:
        fig.update_scenes(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        )

    return fig


def plot_transformation(
    combat: ComBat,
    X: ArrayLike,
    *,
    reduction_method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: Literal[2, 3] = 2,
    plot_type: Literal["static", "interactive"] = "static",
    figsize: tuple[int, int] = (12, 5),
    alpha: float = 0.7,
    point_size: int = 50,
    cmap: str = "Set1",
    title: str | None = None,
    show_legend: bool = True,
    return_embeddings: bool = False,
    **reduction_kwargs: Any,
) -> Any | tuple[Any, dict[str, FloatArray]]:
    """
    Visualize the ComBat transformation effect using dimensionality reduction.

    It shows a before/after comparison of data transformed by `ComBat` using
    PCA, t-SNE, or UMAP to reduce dimensions for visualization.

    Parameters
    ----------
    combat : ComBat
        A fitted ``ComBat`` instance.
    X : array-like of shape (n_samples, n_features)
        Input data to transform and visualize.

    reduction_method : {`'pca'`, `'tsne'`, `'umap'`}, default=`'pca'`
        Dimensionality reduction method.

    n_components : {2, 3}, default=2
        Number of components for dimensionality reduction.

    plot_type : {`'static'`, `'interactive'`}, default=`'static'`
        Visualization type:
        - `'static'`: matplotlib plots (can be saved as images)
        - `'interactive'`: plotly plots (explorable, requires plotly)

    figsize : tuple of int, default=(12, 5)
        Figure size in inches (width, height). Only used for static plots.

    alpha : float, default=0.7
        Marker transparency. Only used for static plots.

    point_size : int, default=50
        Marker size. Only used for static plots.

    cmap : str, default='Set1'
        Matplotlib colormap name for batch colors.

    title : str or None, default=None
        Custom figure title. If None, a default title is generated.

    show_legend : bool, default=True
        Whether to display the batch legend.

    return_embeddings : bool, default=False
        If `True`, return embeddings along with the plot.

    **reduction_kwargs : dict
        Additional keyword arguments passed to the reduction method
        (e.g., ``perplexity`` for t-SNE, ``n_neighbors`` for UMAP).

    Returns
    -------
    fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure object containing the plots.

    embeddings : dict, optional
        If `return_embeddings=True`, dictionary with:
        - `'original'`: embedding of original data
        - `'transformed'`: embedding of ComBat-transformed data
    """
    if not hasattr(combat, "_model") or not hasattr(combat._model, "_gamma_star"):
        raise ValueError(
            "This ComBat instance is not fitted yet. Call 'fit' before 'plot_transformation'."
        )

    if n_components not in [2, 3]:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")
    if reduction_method not in ["pca", "tsne", "umap"]:
        raise ValueError(
            f"reduction_method must be 'pca', 'tsne', or 'umap', got '{reduction_method}'"
        )
    if plot_type not in ["static", "interactive"]:
        raise ValueError(f"plot_type must be 'static' or 'interactive', got '{plot_type}'")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    idx = X.index
    batch_vec_raw = _subset(combat.batch, idx)
    if batch_vec_raw is None:
        raise ValueError("Batch information is required for visualization")
    batch_vec: pd.Series = (
        batch_vec_raw if isinstance(batch_vec_raw, pd.Series) else batch_vec_raw.iloc[:, 0]
    )

    X_transformed = combat.transform(X)

    X_np = X.values
    X_trans_np = X_transformed.values

    if reduction_method == "pca":
        from sklearn.decomposition import PCA

        reducer_orig = PCA(n_components=n_components, **reduction_kwargs)
        reducer_trans = PCA(n_components=n_components, **reduction_kwargs)
    elif reduction_method == "tsne":
        from sklearn.manifold import TSNE

        tsne_params = {"perplexity": 30, "max_iter": 1000, "random_state": 42}
        tsne_params.update(reduction_kwargs)
        reducer_orig = TSNE(n_components=n_components, **tsne_params)
        reducer_trans = TSNE(n_components=n_components, **tsne_params)
    else:
        import umap as umap_module

        umap_params = {"random_state": 42}
        umap_params.update(reduction_kwargs)
        reducer_orig = umap_module.UMAP(n_components=n_components, **umap_params)
        reducer_trans = umap_module.UMAP(n_components=n_components, **umap_params)

    X_embedded_orig = reducer_orig.fit_transform(X_np)
    X_embedded_trans = reducer_trans.fit_transform(X_trans_np)

    if plot_type == "static":
        fig = _create_static_plot(
            X_embedded_orig,
            X_embedded_trans,
            batch_vec,
            reduction_method,
            n_components,
            figsize,
            alpha,
            point_size,
            cmap,
            title,
            show_legend,
        )
    else:
        fig = _create_interactive_plot(
            X_embedded_orig,
            X_embedded_trans,
            batch_vec,
            reduction_method,
            n_components,
            cmap,
            title,
            show_legend,
        )

    if return_embeddings:
        embeddings = {"original": X_embedded_orig, "transformed": X_embedded_trans}
        return fig, embeddings
    else:
        return fig


def plot_feature_diagnostics(
    combat: ComBat,
    top_n: int = 20,
    kind: Literal["location", "scale", "combined"] = "combined",
    mode: Literal["magnitude", "distribution"] = "magnitude",
    layout: Literal["grouped", "diverging"] = "grouped",
    weighted: bool = True,
    figsize: tuple[int, int] = (8, 10),
) -> Any:
    """Plot top features affected by batch effects.

    Parameters
    ----------
    combat : ComBat
        A fitted ``ComBat`` instance.
    top_n : int, default=20
        Number of top features to display.
    kind : {'location', 'scale', 'combined'}, default='combined'
        - 'location': bar plot of location (mean shift) contribution only
        - 'scale': bar plot of scale (variance) contribution only
        - 'combined': grouped bar plot showing location and scale
          side-by-side for each feature (sorted by Euclidean magnitude).
          In magnitude mode: bars reflect Euclidean decomposition
          (combined**2 = location**2 + scale**2).
          In distribution mode: bars reflect independent normalized
          contributions (each sums to 1 separately).
    mode : {'magnitude', 'distribution'}, default='magnitude'
        - 'magnitude': y-axis shows absolute batch effect magnitude
        - 'distribution': y-axis shows relative contribution (proportion), includes
          annotation showing cumulative contribution of top_n features
          (e.g., "Top 20 features explain 75% of total batch effect")
    layout : {'grouped', 'diverging'}, default='grouped'
        Only used when ``kind='combined'``.

        - 'grouped': location and scale bars side-by-side on a single shared
          x-axis.
        - 'diverging': back-to-back bars with location growing leftward and
          scale growing rightward, sharing the feature axis but with an
          independent x-axis and grid per side. Keeps the absolute (or
          relative) values of ``mode`` while giving each component its own
          scale, so a small component is not visually crushed by a large one.
    weighted : bool, default=True
        If True, batch effects are weighted by batch sample size.
        Passed to :func:`feature_batch_diagnostics`.
    figsize : tuple, default=(8,10)
        Figure size (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the plot.

    Raises
    ------
    ValueError
        If the model is not fitted, or if kind/mode is invalid.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if not hasattr(combat, "_model") or not hasattr(combat._model, "_gamma_star"):
        raise ValueError(
            "This ComBat instance is not fitted yet. Call 'fit' before 'plot_feature_diagnostics'."
        )

    if kind not in ["location", "scale", "combined"]:
        raise ValueError(f"kind must be 'location', 'scale', or 'combined', got '{kind}'")

    if mode not in ["magnitude", "distribution"]:
        raise ValueError(f"mode must be 'magnitude' or 'distribution', got '{mode}'")

    if layout not in ["grouped", "diverging"]:
        raise ValueError(f"layout must be 'grouped' or 'diverging', got '{layout}'")

    if layout == "diverging" and kind != "combined":
        raise ValueError("layout='diverging' is only available for kind='combined'")

    importance_df = feature_batch_diagnostics(combat, mode=mode, weighted=weighted)
    top_features = importance_df.head(top_n)

    # Reverse so highest values are at the top of the horizontal bar plot
    top_features = top_features.iloc[::-1]

    if mode == "magnitude":
        value_label = "Magnitude (RMS)"
        title_str = f"Top {top_n} Features by Batch Effect"
    else:
        value_label = "Relative Contribution"
        title_str = f"Top {top_n} Features by Batch Effect (Distribution)"

    if kind == "combined" and layout == "diverging":
        # Back-to-back bars: location grows leftward, scale grows rightward,
        # sharing the feature axis but with an independent x-axis/grid per side.
        fig, (ax_loc, ax_scale) = plt.subplots(
            1, 2, sharey=True, figsize=figsize, gridspec_kw={"wspace": 0}
        )
        y = np.arange(len(top_features))

        ax_loc.barh(
            y, top_features["location"], color="steelblue", edgecolor="black", linewidth=0.5
        )
        ax_loc.set_yticks(y)
        ax_loc.set_yticklabels(top_features.index)
        ax_loc.invert_xaxis()
        ax_loc.set_title("Location")
        ax_loc.set_xlabel(value_label)
        ax_loc.set_ylabel("Feature")

        ax_scale.barh(y, top_features["scale"], color="coral", edgecolor="black", linewidth=0.5)
        ax_scale.set_title("Scale")
        ax_scale.set_xlabel(value_label)

        for sub_ax in (ax_loc, ax_scale):
            sub_ax.grid(axis="x", linestyle=":", alpha=0.6)
            sub_ax.set_axisbelow(True)
            # Blank the zero tick: both panels share it at the center line,
            # so the location and scale labels would otherwise overlap there.
            sub_ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _pos: "" if abs(value) < 1e-9 else f"{value:g}")
            )

        fig.suptitle(title_str + " (Location & Scale)")
    else:
        fig, ax = plt.subplots(figsize=figsize)

        if kind == "combined":
            # Grouped horizontal bar plot showing location and scale side-by-side
            y = np.arange(len(top_features))
            height = 0.35

            ax.barh(
                y + height / 2,
                top_features["location"],
                height,
                label="Location",
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.barh(
                y - height / 2,
                top_features["scale"],
                height,
                label="Scale",
                color="coral",
                edgecolor="black",
                linewidth=0.5,
            )

            ax.set_yticks(y)
            ax.set_yticklabels(top_features.index)
            ax.legend()
        else:
            # Single horizontal bar plot for location or scale
            color = "steelblue" if kind == "location" else "coral"
            ax.barh(
                range(len(top_features)),
                top_features[kind],
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features.index)

        ax.set_ylabel("Feature")
        if mode == "magnitude":
            ax.set_xlabel("Batch Effect Magnitude (RMS)")
        else:
            ax.set_xlabel("Relative Contribution")

        if kind == "combined":
            title_str += " (Location & Scale)"
        else:
            title_str += f" ({kind.capitalize()})"

        ax.set_title(title_str)

    plt.tight_layout()

    # For distribution mode, print cumulative contribution
    if mode == "distribution":
        if kind == "combined":
            cumulative_pct = top_features["combined"].sum() * 100
            effect_label = "batch effect"
        elif kind == "location":
            cumulative_pct = top_features["location"].sum() * 100
            effect_label = "location effect"
        else:  # scale
            cumulative_pct = top_features["scale"].sum() * 100
            effect_label = "scale effect"

        print(f"Top {top_n} features explain {cumulative_pct:.1f}% of total {effect_label}")

    return fig


def plot_batch_effect_heatmap(
    combat: ComBat,
    top_n: int = 50,
    weighted: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> Any:
    """Plot a heatmap of batch effect parameters across features and batches.

    Displays the estimated batch-specific location shifts (gamma) and,
    unless ``mean_only=True``, log-scale shifts (log delta) for the
    ``top_n`` most affected features.

    Parameters
    ----------
    combat : ComBat
        A fitted ``ComBat`` instance.
    top_n : int, default=50
        Number of top features (by combined batch effect) to display.
    weighted : bool, default=True
        If True, feature ranking uses sample-size-weighted batch effects.
        Passed to :func:`feature_batch_diagnostics`.
    figsize : tuple of int, default=(12, 8)
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the heatmap(s).

    Raises
    ------
    ValueError
        If the model is not fitted.
    ImportError
        If seaborn is not installed.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not hasattr(combat, "_model") or not hasattr(combat._model, "_gamma_star"):
        raise ValueError(
            "This ComBat instance is not fitted yet. Call 'fit' before 'plot_batch_effect_heatmap'."
        )

    feature_names = combat._model._grand_mean.index
    batch_levels = combat._model._batch_levels
    gamma_star = combat._model._gamma_star
    delta_star = combat._model._delta_star

    importance = feature_batch_diagnostics(combat, weighted=weighted)
    top_features = importance.head(top_n).index
    feat_idx = np.array([feature_names.get_loc(f) for f in top_features])

    gamma_df = pd.DataFrame(
        gamma_star[:, feat_idx],
        index=[str(b) for b in batch_levels],
        columns=top_features,
    )

    n_plots = 1 if combat.mean_only else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    sns.heatmap(
        gamma_df,
        cmap="RdBu_r",
        center=0,
        ax=axes[0],
        xticklabels=True,
        yticklabels=True,
    )
    axes[0].set_title("Location shifts (gamma)")
    axes[0].set_xlabel("Feature")
    axes[0].set_ylabel("Batch")

    if not combat.mean_only:
        log_delta_df = pd.DataFrame(
            np.log(delta_star[:, feat_idx]),
            index=[str(b) for b in batch_levels],
            columns=top_features,
        )
        sns.heatmap(
            log_delta_df,
            cmap="RdBu_r",
            center=0,
            ax=axes[1],
            xticklabels=True,
            yticklabels=True,
        )
        axes[1].set_title("Scale shifts (log delta)")
        axes[1].set_xlabel("Feature")
        axes[1].set_ylabel("Batch")

    fig.suptitle(f"Batch Effect Heatmap (top {top_n} features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig

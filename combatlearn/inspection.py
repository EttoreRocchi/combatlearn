"""Standalone inspection functions for fitted ComBat models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .sklearn_api import ComBat


def feature_batch_diagnostics(
    combat: ComBat,
    mode: Literal["magnitude", "distribution"] = "magnitude",
    weighted: bool = True,
) -> pd.DataFrame:
    """Compute per-feature batch effect magnitude.

    Returns a DataFrame with columns ``location``, ``scale``, and
    ``combined``. Location is the (weighted) RMS of gamma across batches
    (standardized mean shifts). Scale is the (weighted) RMS of log-delta
    across batches (log-fold variance change). Combined is the Euclidean
    norm sqrt(location**2 + scale**2). Using RMS provides L2-consistent
    aggregation; using log(delta) ensures symmetry.

    Parameters
    ----------
    combat : ComBat
        A fitted ``ComBat`` instance.
    mode : {'magnitude', 'distribution'}, default='magnitude'
        - 'magnitude': Returns L2-consistent absolute batch effect magnitudes.
          Suitable for ranking, thresholding, and cross-dataset comparison.
        - 'distribution': Returns column-wise normalized proportions (each column
          sums to 1, values in range [0, 1]), representing the relative contribution
          of each feature to the total location, scale, or combined batch effect.
          Note: normalization is applied independently to each column, so the
          Euclidean relationship (combined**2 = location**2 + scale**2) no longer holds.
    weighted : bool, default=True
        If True, compute a weighted RMS where each batch is weighted by its
        sample size. This gives more influence to larger batches, producing
        a more statistically representative summary. If False, all batches
        contribute equally regardless of size.

    Returns
    -------
    pd.DataFrame
        DataFrame with index=feature names, columns=['location', 'scale', 'combined'],
        sorted by 'combined' descending.

    Raises
    ------
    ValueError
        If the model is not fitted or if mode is invalid.
    """
    if not hasattr(combat, "_model") or not hasattr(combat._model, "_gamma_star"):
        raise ValueError(
            "This ComBat instance is not fitted yet. "
            "Call 'fit' before 'feature_batch_diagnostics'."
        )

    if mode not in ["magnitude", "distribution"]:
        raise ValueError(f"mode must be 'magnitude' or 'distribution', got '{mode}'")

    feature_names = combat._model._grand_mean.index
    gamma_star = combat._model._gamma_star
    delta_star = combat._model._delta_star

    if weighted:
        # Batch sample sizes as weights, normalized to sum to 1
        n_per_batch = combat._model._n_per_batch
        weights = np.array(
            [n_per_batch[str(lvl)] for lvl in combat._model._batch_levels], dtype=np.float64
        )
        weights = weights / weights.sum()

        # Location effect: weighted RMS of gamma across batches
        location = np.sqrt((weights[:, np.newaxis] * gamma_star**2).sum(axis=0))

        # Scale effect: weighted RMS of log(delta) across batches
        if not combat.mean_only:
            scale = np.sqrt((weights[:, np.newaxis] * np.log(delta_star) ** 2).sum(axis=0))
        else:
            scale = np.zeros_like(location)
    else:
        # Location effect: unweighted RMS of gamma across batches
        location = np.sqrt((gamma_star**2).mean(axis=0))

        # Scale effect: unweighted RMS of log(delta) across batches
        if not combat.mean_only:
            scale = np.sqrt((np.log(delta_star) ** 2).mean(axis=0))
        else:
            scale = np.zeros_like(location)

    # Euclidean to treat location and scale as orthogonal dimensions
    combined = np.sqrt(location**2 + scale**2)

    if mode == "distribution":
        # Normalize each column independently to sum to 1
        location_sum = location.sum()
        scale_sum = scale.sum()
        combined_sum = combined.sum()

        location = location / location_sum if location_sum > 0 else location
        scale = scale / scale_sum if scale_sum > 0 else scale
        combined = combined / combined_sum if combined_sum > 0 else combined

    return pd.DataFrame(
        {
            "location": location,
            "scale": scale,
            "combined": combined,
        },
        index=feature_names,
    ).sort_values("combined", ascending=False)


def summary(combat: ComBat) -> str:
    """Return a human-readable diagnostic report after fitting.

    Parameters
    ----------
    combat : ComBat
        A fitted ``ComBat`` instance.

    Returns
    -------
    str
        Multi-line summary string.

    Raises
    ------
    ValueError
        If the model is not fitted.
    """
    if not hasattr(combat, "_model") or not hasattr(combat._model, "_gamma_star"):
        raise ValueError("This ComBat instance is not fitted yet. Call 'fit' before 'summary'.")

    lines: list[str] = []
    lines.append("ComBat Summary")
    lines.append("=" * 40)
    lines.append(f"Method:          {combat.method}")
    lines.append(f"Parametric:      {combat.parametric}")
    lines.append(f"Mean only:       {combat.mean_only}")
    lines.append(f"Reference batch: {combat.reference_batch or 'None'}")

    batch_levels = combat._model._batch_levels
    n_per_batch = combat._model._n_per_batch
    lines.append(f"Number of batches: {len(batch_levels)}")
    lines.append("Samples per batch:")
    for lvl in batch_levels:
        lines.append(f"  {lvl}: {n_per_batch[str(lvl)]}")

    n_features = len(combat._model._grand_mean)
    lines.append(f"Number of features: {n_features}")

    lines.append("")
    lines.append("Top 5 features by batch effect (combined):")
    importance = feature_batch_diagnostics(combat)
    top5 = importance.head(5)
    for feat, row in top5.iterrows():
        lines.append(f"  {feat}: {row['combined']:.4f}")

    # Diagnostics table
    lines.append("")
    lines.append("Diagnostics")
    lines.append("=" * 40)
    col_w = 34
    lines.append(f"{'Metric':<{col_w}}Value")
    lines.append(f"{'------':<{col_w}}-----")

    if hasattr(combat, "_batch_var_before_"):
        lines.append(f"{'Batch var. explained (before)':<{col_w}}{combat._batch_var_before_:.1%}")
    if hasattr(combat, "_batch_var_after_"):
        lines.append(f"{'Batch var. explained (after)':<{col_w}}{combat._batch_var_after_:.1%}")

    design_cond = getattr(combat._model, "_design_cond", None)
    if design_cond is not None:
        lines.append(f"{'Design matrix condition number':<{col_w}}{design_cond:.1f}")

    conv_info = getattr(combat._model, "_convergence_info", [])
    if conv_info:
        eb_type = "parametric" if combat.parametric else "non-parametric"
        lines.append(f"EB convergence ({eb_type}):")
        for info in conv_info:
            batch_name = info["batch"]
            if info["converged"]:
                status = f"converged ({info['iterations']} iter)"
            else:
                max_change = max(info["final_gamma_change"], info["final_delta_change"])
                status = f"NOT CONVERGED ({info['iterations']} iter, \u0394={max_change:.2e})"
            lines.append(f"  {batch_name!s:<{col_w - 2}}{status}")

    return "\n".join(lines)

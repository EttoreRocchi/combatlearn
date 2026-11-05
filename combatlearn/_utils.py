"""Internal utilities shared across combatlearn modules."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .core import ArrayLike


def _fit_levels(values: pd.Series) -> pd.Index:
    """Levels a discrete covariate takes in the fitted rows (unused categories dropped)."""
    levels: pd.Index = pd.Index(pd.Categorical(values).remove_unused_categories().categories)
    return levels


def _code_levels(values: pd.Series, levels: pd.Index, label: str) -> pd.Categorical:
    """Code values against the levels frozen at fit; a level not seen at fit raises.

    Dummy coding would otherwise turn it into an all-zero row, silently the reference level.
    """
    codes = levels.get_indexer(pd.Index(values))
    unseen = pd.unique(values[(codes == -1) & values.notna().to_numpy()])
    if len(unseen):
        raise ValueError(
            f"{label} has level(s) {sorted(map(str, unseen))} that were "
            f"not seen at fit. A level with no training samples has no estimated covariate "
            f"effect, so it cannot be corrected; coding it as the reference level would "
            f"silently apply the wrong one. Seen at fit: {sorted(map(str, levels))}."
        )
    return pd.Categorical.from_codes(codes.tolist(), categories=levels)


def _check_positional_alignment(
    X: ArrayLike, auxiliaries: Iterable[tuple[str, ArrayLike | None]]
) -> None:
    """Require length-matched auxiliaries when ``X`` carries no index labels.

    A plain-array ``X`` can only align positionally, which breaks inside CV when the
    splitter subsets ``X`` but construction-time ``batch``/covariates stay full-length.
    """
    if isinstance(X, pd.DataFrame):
        return
    n = len(X)
    for name, obj in auxiliaries:
        if obj is None:
            continue
        if len(obj) != n:
            raise ValueError(
                f"`{name}` has length {len(obj)} but X has {n} rows. When X is not a "
                f"pandas DataFrame it has no index labels, so combatlearn can only align "
                f"`{name}` to X positionally, which requires equal lengths. This typically "
                f"happens inside cross-validation, where the splitter subsets X but the "
                f"batch/covariates passed at construction stay full-length. Pass X and "
                f"`{name}` as pandas objects sharing the same index so folds stay "
                f"label-aligned, or pass a same-length `{name}`."
            )


def _subset(obj: ArrayLike | None, idx: pd.Index) -> pd.DataFrame | pd.Series | None:
    """Subset array-like object by index."""
    if obj is None:
        return None
    if isinstance(obj, pd.Series | pd.DataFrame):
        return obj.loc[idx]
    else:
        if isinstance(obj, np.ndarray) and obj.ndim == 1:
            return pd.Series(obj, index=idx)
        else:
            return pd.DataFrame(obj, index=idx)


def _warn_if_covariate_determines_target(
    y: ArrayLike | None,
    discrete_covariates: pd.DataFrame | pd.Series | None,
    *,
    estimator: str,
    stacklevel: int = 4,
) -> None:
    """Warn when a discrete covariate perfectly determines ``y`` (likely the target).

    Protecting the target inflates cross-validated scores. ``discrete_covariates``
    must already be subset to the fitted rows; covariates averaging fewer than two
    samples per level are skipped to avoid false alarms.
    """
    if y is None or discrete_covariates is None:
        return

    target = pd.Series(np.asarray(y).ravel())
    if len(target) != len(discrete_covariates) or target.nunique(dropna=False) < 2:
        return

    frame = (
        discrete_covariates.to_frame()
        if isinstance(discrete_covariates, pd.Series)
        else discrete_covariates
    )
    n_samples = len(frame)
    for column in frame.columns:
        levels = pd.Series(np.asarray(frame[column]))
        n_levels = levels.nunique(dropna=False)
        if n_levels < 2 or n_levels * 2 > n_samples:
            continue
        if target.groupby(list(levels), dropna=False).nunique(dropna=False).max() > 1:
            continue
        warnings.warn(
            f"{estimator}: discrete covariate {column!r} determines `y` on the training "
            f"data, so this correction preserves the very signal you are about to predict, "
            f"and `transform` applies it using each held-out sample's own label - which "
            f"would not be available at prediction time. Any cross-validated score will be "
            f"optimistic. If you are harmonizing a whole cohort for differential expression "
            f"or methylation, use combatlearn.transductive.TransductiveComBat(condition=...), "
            f"which protects the term and refuses `transform`. If {column!r} is genuinely "
            f"known at intake and is not your target, this warning does not apply.",
            UserWarning,
            stacklevel=stacklevel,
        )

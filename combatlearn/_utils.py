"""Internal utilities shared across combatlearn modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .core import ArrayLike


def _check_positional_alignment(
    X: ArrayLike, auxiliaries: Iterable[tuple[str, ArrayLike | None]]
) -> None:
    """Require length-matched auxiliaries when ``X`` carries no index labels.

    ``batch`` and the covariates are supplied at construction and label-aligned to
    ``X`` via its pandas index. When ``X`` is a plain array it has no labels, so the
    only possible alignment is positional, which needs equal lengths. This most
    often breaks inside cross-validation: the CV splitter subsets ``X`` positionally
    while the construction-time ``batch``/covariates stay full-length, and the
    silent length mismatch would otherwise surface as a confusing pandas reindexing
    error. A DataFrame ``X`` label-aligns and is skipped.

    Parameters
    ----------
    X : array-like
        The data passed to ``fit``/``transform``.
    auxiliaries : iterable of (name, object)
        The construction-time vectors to check; ``None`` entries are ignored.

    Raises
    ------
    ValueError
        If ``X`` is not a DataFrame and any auxiliary has a different length.
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

"""Internal utilities shared across combatlearn modules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import ArrayLike


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

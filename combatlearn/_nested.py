"""Orchestration helpers for NestedComBat (*Horng et al.* 2022).

The sequential single-batch ComBat fits, the Anderson-Darling order-search
objective, and the GMM grouping; all are learned at fit and frozen for transform.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp
from sklearn.mixture import GaussianMixture

from .core import ComBatModel

GMM_VAR_NAME = "GMM"

ModelStep = tuple[str, ComBatModel]


def variance_explained_by_batch(X: pd.DataFrame, batch: pd.Series) -> float:
    """Fraction of total variance explained by ``batch`` (summed across features)."""
    values = X.to_numpy(dtype=np.float64)
    labels = np.asarray(batch)
    grand = values.mean(axis=0)
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = 0.0
    for lvl in pd.unique(labels):
        mask = labels == lvl
        batch_mean = values[mask].mean(axis=0)
        ss_between += float(mask.sum()) * float(((batch_mean - grand) ** 2).sum())
    return ss_between / ss_total


def anderson_darling_batch_count(
    X: pd.DataFrame,
    batch_vars: Mapping[str, pd.Series],
    *,
    alpha: float = 0.05,
) -> int:
    """Number of features with a significant residual batch effect.

    Counts (feature, batch variable) pairs whose Anderson-Darling k-sample test
    (:func:`scipy.stats.anderson_ksamp`) is significant at ``alpha``; degenerate
    features do not count. This is the objective the OPNested order search minimizes.

    Parameters
    ----------
    X : pd.DataFrame of shape (n_samples, n_features)
        Harmonized data to score.
    batch_vars : mapping of str to pd.Series
        Batch variables, each a per-sample label vector.
    alpha : float, default=0.05
        Significance threshold for the Anderson-Darling test.

    Returns
    -------
    int
        Total number of significant (feature, batch variable) pairs.
    """
    col_values = X.to_numpy(dtype=np.float64)
    n_features = col_values.shape[1]
    total = 0
    with warnings.catch_warnings():
        # Warns only when p is capped to [0.001, 0.25]; harmless at alpha=0.05.
        warnings.simplefilter("ignore")
        for batch in batch_vars.values():
            labels = np.asarray(batch)
            masks = [labels == lvl for lvl in pd.unique(labels)]
            masks = [m for m in masks if m.any()]
            if len(masks) < 2:
                continue
            for j in range(n_features):
                col = col_values[:, j]
                samples = [col[m] for m in masks]
                try:
                    result = anderson_ksamp(samples)
                except ValueError:
                    continue
                if result.significance_level < alpha:
                    total += 1
    return total


def fit_nested_sequence(
    order: Sequence[str],
    X: pd.DataFrame,
    batch_vars: Mapping[str, pd.Series],
    disc: pd.DataFrame | None,
    cont: pd.DataFrame | None,
    make_model: Callable[[str], ComBatModel],
) -> tuple[list[ModelStep], pd.DataFrame]:
    """Fit one ``ComBatModel`` per batch variable in ``order``, chaining transforms.

    Covariates are protected at every step. Returns the ``(batch_variable, model)``
    steps and the fully harmonized training data.
    """
    models: list[ModelStep] = []
    X_current = X
    for name in order:
        model = make_model(name)
        batch = batch_vars[name]
        model.fit(X_current, batch=batch, discrete_covariates=disc, continuous_covariates=cont)
        X_current = model.transform(
            X_current, batch=batch, discrete_covariates=disc, continuous_covariates=cont
        )
        models.append((name, model))
    return models, X_current


def transform_nested_sequence(
    models: Sequence[ModelStep],
    X: pd.DataFrame,
    batch_vars: Mapping[str, pd.Series],
    disc: pd.DataFrame | None,
    cont: pd.DataFrame | None,
) -> pd.DataFrame:
    """Replay a fitted nested sequence on new data, in the stored order."""
    X_current = X
    for name, model in models:
        batch = batch_vars[name]
        X_current = model.transform(
            X_current, batch=batch, discrete_covariates=disc, continuous_covariates=cont
        )
    return X_current


def _search_exhaustive(
    names: Sequence[str],
    X: pd.DataFrame,
    batch_vars: Mapping[str, pd.Series],
    disc: pd.DataFrame | None,
    cont: pd.DataFrame | None,
    make_model: Callable[[str], ComBatModel],
    alpha: float,
) -> tuple[list[str], list[ModelStep]]:
    """Try every ordering, keeping the first that minimizes the AD feature count."""
    best_score: int | None = None
    best_order: list[str] = list(names)
    best_models: list[ModelStep] = []
    for order in permutations(names):
        models, X_harmonized = fit_nested_sequence(order, X, batch_vars, disc, cont, make_model)
        score = anderson_darling_batch_count(X_harmonized, batch_vars, alpha=alpha)
        if best_score is None or score < best_score:
            best_score = score
            best_order = list(order)
            best_models = models
    return best_order, best_models


def _search_greedy(
    names: Sequence[str],
    X: pd.DataFrame,
    batch_vars: Mapping[str, pd.Series],
    disc: pd.DataFrame | None,
    cont: pd.DataFrame | None,
    make_model: Callable[[str], ComBatModel],
    alpha: float,
) -> tuple[list[str], list[ModelStep]]:
    """Greedy forward selection: repeatedly apply the step that most lowers the AD count."""
    remaining = list(names)
    order: list[str] = []
    models: list[ModelStep] = []
    X_current = X
    while remaining:
        best_score: int | None = None
        best_name = remaining[0]
        best_model: ComBatModel | None = None
        best_X = X_current
        for name in remaining:
            model = make_model(name)
            batch = batch_vars[name]
            model.fit(X_current, batch=batch, discrete_covariates=disc, continuous_covariates=cont)
            X_try = model.transform(
                X_current, batch=batch, discrete_covariates=disc, continuous_covariates=cont
            )
            score = anderson_darling_batch_count(X_try, batch_vars, alpha=alpha)
            if best_score is None or score < best_score:
                best_score = score
                best_name = name
                best_model = model
                best_X = X_try
        assert best_model is not None
        order.append(best_name)
        models.append((best_name, best_model))
        X_current = best_X
        remaining.remove(best_name)
    return order, models


def search_order(
    names: Sequence[str],
    X: pd.DataFrame,
    batch_vars: Mapping[str, pd.Series],
    disc: pd.DataFrame | None,
    cont: pd.DataFrame | None,
    make_model: Callable[[str], ComBatModel],
    *,
    max_exhaustive_vars: int,
    alpha: float = 0.05,
) -> tuple[list[str], list[ModelStep], bool]:
    """Select a harmonization order that minimizes the residual batch effect.

    Exhaustive over all ``k!`` orderings when ``k <= max_exhaustive_vars`` (warning
    from ``k >= 4``), else greedy with a warning. Returns the order, its fitted
    sequence, and whether the greedy fallback was used.
    """
    k = len(names)
    if k <= max_exhaustive_vars:
        if k >= 4:
            n_orderings = factorial(k)
            warnings.warn(
                f"OPNested exhaustive order search over {k} batch variables will fit "
                f"{n_orderings} orderings x {k} steps = {n_orderings * k} ComBat models, each "
                f"scored with a per-feature Anderson-Darling test; on high-dimensional data "
                f"this can be slow. Lower max_exhaustive_vars to use the greedy search instead, "
                f"or set optimize_order=False to skip the search.",
                UserWarning,
                stacklevel=2,
            )
        order, models = _search_exhaustive(names, X, batch_vars, disc, cont, make_model, alpha)
        return order, models, False
    warnings.warn(
        f"{len(names)} batch variables exceed max_exhaustive_vars="
        f"{max_exhaustive_vars}; an exhaustive search over {len(names)}! orderings is "
        f"impractical, so falling back to greedy forward selection. Raise "
        f"max_exhaustive_vars to force the exhaustive search.",
        UserWarning,
        stacklevel=2,
    )
    order, models = _search_greedy(names, X, batch_vars, disc, cont, make_model, alpha)
    return order, models, True


class GMMGrouping:
    """A per-feature Gaussian-mixture grouping (*Horng et al.* 2022, GMM ComBat).

    A 2-component mixture is fit to each feature; among those whose smaller cluster
    exceeds ``min_cluster_frac`` of the samples, the lowest-AIC one gives the grouping,
    used as a batch variable (``+GMM``) or covariate (``-GMM``).
    """

    def __init__(self, feature: object, gmm: GaussianMixture) -> None:
        self.feature = feature
        self.gmm = gmm

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        *,
        min_cluster_frac: float = 0.25,
        random_state: int | None = 0,
    ) -> GMMGrouping | None:
        """Fit the grouping, or return ``None`` if no feature yields a balanced split."""
        n_samples = len(X)
        best_feature: object = None
        best_aic = np.inf
        best_gmm: GaussianMixture | None = None
        for col in X.columns:
            x = X[col].to_numpy(dtype=np.float64).reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, random_state=random_state)
            try:
                gmm.fit(x)
                labels = gmm.predict(x)
            except ValueError:
                continue
            smaller = min(int((labels == 0).sum()), int((labels == 1).sum()))
            if smaller <= min_cluster_frac * n_samples:
                continue
            aic = float(gmm.aic(x))
            if aic < best_aic:
                best_aic = aic
                best_feature = col
                best_gmm = gmm
        if best_gmm is None:
            return None
        return cls(best_feature, best_gmm)

    def assign(self, X: pd.DataFrame) -> pd.Series:
        """Assign each sample to its mixture component (``'gmm0'`` / ``'gmm1'``)."""
        x = X[self.feature].to_numpy(dtype=np.float64).reshape(-1, 1)
        labels = self.gmm.predict(x)
        return pd.Series([f"gmm{int(c)}" for c in labels], index=X.index, name=GMM_VAR_NAME)

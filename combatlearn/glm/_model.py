"""Fit/transform engine for the GLM + quantile-map ComBat family.

Shared by :class:`~combatlearn.glm.GLMComBat` and
:class:`~combatlearn.transductive.TransductiveComBat`. The fit stores only
per-feature and per-batch constants, so held-out samples are harmonized one at a time.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
import pandas as pd

from .._utils import _code_levels, _fit_levels
from ..core import ArrayLike, FloatArray, _resolve_method
from ._beta import beta_glm_fit, common_precision, featurewise_precision
from ._nb import common_dispersion, featurewise_dispersion, nb_glm_fit
from ._quantile import map_to_batch_free_beta, map_to_batch_free_nb

# Reference value: keeps the coefficients finite when a design cell is empty
_PRIOR_COUNT = 1e-4

_METHODS = ("seq", "met")


def _expit(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def _has_residual_signal(
    values: FloatArray, design: FloatArray, tol: float = 1e-8
) -> npt.NDArray[np.bool_]:
    """Per-feature test that ``values`` is not an exact linear function of ``design``.

    Such a feature has no residual variation, so its beta regression is unidentifiable.
    """
    fitted = design @ la.lstsq(design, values.T, rcond=None)[0]
    residual_ss = ((values.T - fitted) ** 2).sum(axis=0)
    total_ss = ((values - values.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return np.asarray(residual_ss > tol * np.maximum(total_ss, 1.0))


class GLMComBatModel:
    """Non-Gaussian GLM ComBat engine (ComBat-seq and ComBat-met).

    Parameters
    ----------
    method : {'seq', 'met'}, default='seq'
        * ``'seq'`` - ComBat-seq (Zhang, Parmigiani & Johnson 2020): a
          negative-binomial GLM with a per-sample library-size offset, mapped
          back through the NB quantile function so the output stays integral.
        * ``'met'`` - ComBat-met (Wang 2025): a logit-link beta regression for
          proportions, mapped through the beta quantile function.
    reference_batch : str, optional
        Batch level whose distribution becomes the correction target; its
        samples are returned unchanged. It pins the target mean, and for
        ``'met'`` also the target precision.
    min_samples_per_batch : int, default=2
        Smallest batch accepted at fit. Per-batch dispersions cannot be
        estimated from a single sample.
    pseudo_beta : float, default=1e-4
        ``'met'`` only: value substituted for observations at exactly 0 or 1,
        which lie outside the support of the beta distribution. Must be in
        ``(0, 0.5)``.
    """

    def __init__(
        self,
        *,
        method: Literal["seq", "met"] = "seq",
        reference_batch: str | None = None,
        min_samples_per_batch: int = 2,
        pseudo_beta: float = 1e-4,
    ) -> None:
        self.method = method
        self.reference_batch = reference_batch
        self.min_samples_per_batch = min_samples_per_batch
        self.pseudo_beta = pseudo_beta

    def _check_method(self) -> str:
        method = _resolve_method(str(self.method))
        if method not in _METHODS:
            raise ValueError(
                f"method={self.method!r} is not recognized. GLMComBat supports "
                f"'seq' (ComBat-seq, negative-binomial counts) and 'met' "
                f"(ComBat-met, beta-distributed proportions)."
            )
        return method

    def _check_values(self, X: pd.DataFrame, method: str) -> FloatArray:
        values = np.asarray(X.values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("X contains NaN or infinite values; GLMComBat requires complete data.")
        if method == "seq":
            if (values < 0).any():
                raise ValueError(
                    "method='seq' expects raw RNA-seq counts, but X contains negative values. "
                    "Do not log-transform, normalize or center the data before ComBat-seq."
                )
            if not np.array_equal(values, np.rint(values)):
                raise ValueError(
                    "method='seq' expects integer counts, but X contains non-integer values. "
                    "Pass raw counts; for continuous data use combatlearn.ComBat instead."
                )
        else:
            if (values < 0).any() or (values > 1).any():
                raise ValueError(
                    "method='met' expects methylation beta-values in [0, 1], but X contains "
                    "values outside that range. Convert M-values with expit(m) first, or use "
                    "combatlearn.ComBat for unbounded data."
                )
            if not 0.0 < self.pseudo_beta < 0.5:
                raise ValueError(
                    f"pseudo_beta={self.pseudo_beta!r} must lie strictly between 0 and 0.5."
                )
            values = np.where(values <= 0.0, self.pseudo_beta, values)
            values = np.where(values >= 1.0, 1.0 - self.pseudo_beta, values)
        return values

    @staticmethod
    def _reject_missing(missing: int, name: str) -> None:
        """Refuse null labels/covariates, at fit and at transform alike.

        A null batch label codes to ``-1`` and would silently index the last batch.
        """
        if missing:
            raise ValueError(
                f"`{name}` contains {missing} NaN value(s). All values must be non-null; "
                f"a missing label cannot be assigned a batch effect. Drop or impute the "
                f"affected samples."
            )

    @classmethod
    def _as_categorical(cls, arr: ArrayLike, index: pd.Index, name: str) -> pd.Series:
        series = (
            arr.copy() if isinstance(arr, pd.Series) else pd.Series(arr, index=index, name=name)
        )
        if not series.index.equals(index):
            raise ValueError(
                f"`{name}` index does not align with `X`. Ensure both share the same index."
            )
        cls._reject_missing(int(series.isna().sum()), name)
        return series.astype("category")

    @classmethod
    def _as_frame(cls, arr: ArrayLike | None, index: pd.Index, name: str) -> pd.DataFrame | None:
        if arr is None:
            return None
        if isinstance(arr, pd.Series):
            arr = arr.to_frame(name=name)
        if not isinstance(arr, pd.DataFrame):
            arr = pd.DataFrame(np.asarray(arr), index=index)
        if not arr.index.equals(index):
            raise ValueError(
                f"`{name}` index does not align with `X`. Ensure both share the same index."
            )
        cls._reject_missing(int(arr.isna().sum().sum()), name)
        return arr

    def _covariate_frame(
        self,
        index: pd.Index,
        discrete: pd.DataFrame | None,
        continuous: pd.DataFrame | None,
        condition: pd.DataFrame | None,
        *,
        fitting: bool,
    ) -> pd.DataFrame:
        """Non-batch design block: condition dummies, then covariates.

        Dummies use the levels recorded at fit, so a single held-out sample is
        encoded as it would be in any larger transform set.
        """
        if fitting:
            self._category_levels: dict[str, pd.Index] = {}

        parts: list[pd.DataFrame] = []
        for block, frame in (("condition", condition), ("covariate", discrete)):
            if frame is None:
                continue
            coded = pd.DataFrame(index=index)
            for column in frame.columns:
                key = f"{block}::{column}"
                if fitting:
                    self._category_levels[key] = _fit_levels(frame[column])
                levels = self._category_levels.get(key)
                if levels is None:
                    raise ValueError(
                        f"`{block}` column {column!r} was not present at fit. Pass the same "
                        f"covariates to transform as to fit."
                    )
                param = "condition" if block == "condition" else "discrete_covariates"
                coded[f"{block}_{column}"] = _code_levels(
                    frame[column], levels, f"{param} column {column!r}"
                )
            parts.append(pd.get_dummies(coded, drop_first=True).astype(float))

        if continuous is not None:
            parts.append(continuous.astype(float))
        if not parts:
            return pd.DataFrame(index=index)
        return pd.concat(parts, axis=1)

    def _batch_dummies(self, batch: pd.Series, method: str) -> FloatArray:
        dummies = (
            pd.get_dummies(batch)
            .reindex(columns=self._batch_levels, fill_value=False)
            .astype(float)
        )
        if method == "seq" and self._reference_index is not None:
            # As in ComBat-seq: the reference column becomes the intercept.
            dummies.iloc[:, self._reference_index] = 1.0
        return np.asarray(dummies.values, dtype=np.float64)

    def _assemble(self, batch: pd.Series, covariates: pd.DataFrame, method: str) -> FloatArray:
        blocks = [self._batch_dummies(batch, method)]
        if covariates.shape[1]:
            blocks.append(np.asarray(covariates.values, dtype=np.float64))
        return np.column_stack(blocks)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        batch: ArrayLike,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        condition: ArrayLike | None = None,
    ) -> GLMComBatModel:
        """Estimate and freeze every parameter the transform needs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Counts (``'seq'``) or beta-values (``'met'``).
        y : None
            Ignored. Present for API compatibility.
        batch : array-like of shape (n_samples,)
            Batch labels.
        discrete_covariates, continuous_covariates : array-like, optional
            Biological signal to preserve.
        condition : array-like, optional
            Condition-of-interest / outcome term to protect. Only the
            transductive wrapper supplies this: it is unavailable at transform
            time, so a model fitted with it can only be applied to its own
            cohort.

        Returns
        -------
        self : GLMComBatModel
        """
        method = self._check_method()
        self._method = method
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        index = X.index
        self._feature_names = X.columns

        batch = self._as_categorical(batch, index, "batch")
        self._batch_levels = batch.cat.categories
        n_batch = len(self._batch_levels)
        if n_batch < 2:
            raise ValueError(
                f"found {n_batch} batch level(s); at least 2 are needed for batch correction."
            )

        counts_per_batch = batch.value_counts().reindex(self._batch_levels).astype(int)
        too_small = counts_per_batch[counts_per_batch < self.min_samples_per_batch]
        if len(too_small):
            raise ValueError(
                f"batches {list(too_small.index)} have fewer than "
                f"{self.min_samples_per_batch} samples "
                f"({dict(too_small)}). Per-batch dispersions cannot be estimated from a "
                f"single sample. Merge the small batches, drop them, or lower "
                f"`min_samples_per_batch` if you accept the added variance."
            )

        self._reference_index = None
        if self.reference_batch is not None:
            if self.reference_batch not in self._batch_levels:
                raise ValueError(
                    f"reference_batch={self.reference_batch!r} not found. "
                    f"Available batches: {list(self._batch_levels)}."
                )
            self._reference_index = int(list(self._batch_levels).index(self.reference_batch))

        discrete = self._as_frame(discrete_covariates, index, "discrete_covariates")
        continuous = self._as_frame(continuous_covariates, index, "continuous_covariates")
        condition_frame = self._as_frame(condition, index, "condition")
        covariates = self._covariate_frame(
            index, discrete, continuous, condition_frame, fitting=True
        )
        self._nonbatch_columns = list(covariates.columns)
        self._has_condition = condition_frame is not None

        values = self._check_values(X, method).T
        design = self._assemble(batch, covariates, method)
        if la.matrix_rank(design) < design.shape[1]:
            raise ValueError(
                "the design matrix is rank-deficient: a covariate (or the condition) is "
                "confounded with batch, so their effects cannot be separated. Drop the "
                "confounded term and refit."
            )
        within = np.column_stack([np.ones(len(index)), design[:, n_batch:]])
        codes = np.asarray(batch.cat.codes.values, dtype=np.intp)
        sizes = np.asarray(counts_per_batch.values, dtype=np.float64)

        self._degenerate_batches: set[int] = set()
        if method == "seq":
            self._fit_seq(values, codes, design, within, sizes)
        else:
            self._fit_met(values, codes, design, within, sizes)
        self._warn_degenerate_batches()
        return self

    def _batch_design(
        self, within: FloatArray, cols: npt.NDArray[np.intp], n_nonbatch: int, batch: int
    ) -> tuple[FloatArray, bool]:
        """Within-batch design, falling back to an intercept when residual df run out."""
        has_df = len(cols) > n_nonbatch + 1 and la.matrix_rank(within[cols]) == within.shape[1]
        if not has_df:
            self._degenerate_batches.add(batch)
        return (within[cols] if has_df else np.ones((len(cols), 1))), has_df

    def _warn_degenerate_batches(self) -> None:
        """Warn about batches whose per-batch fit had to drop the covariates."""
        if not self._degenerate_batches:
            return
        names = sorted(str(self._batch_levels[b]) for b in self._degenerate_batches)
        warnings.warn(
            f"batch(es) {names} have too few samples to support the within-batch design, "
            f"so their per-batch "
            f"{'dispersion' if self._method == 'seq' else 'precision'} is estimated from an "
            f"intercept-only model that ignores the covariates. The correction still runs, "
            f"but the estimate for those batches is cruder. Merge them into a larger batch, "
            f"drop them, or raise `min_samples_per_batch` to reject them outright.",
            UserWarning,
            stacklevel=4,
        )

    def _fit_seq(
        self,
        values: FloatArray,
        codes: npt.NDArray[np.intp],
        design: FloatArray,
        within: FloatArray,
        sizes: FloatArray,
    ) -> None:
        n_batch = len(self._batch_levels)
        n_samples = values.shape[1]
        n_nonbatch = design.shape[1] - n_batch

        keep: npt.NDArray[np.bool_] = np.ones(values.shape[0], dtype=bool)
        for b in range(n_batch):
            keep &= values[:, codes == b].sum(axis=1) > 0
        self._adjustable = keep
        if not keep.any():
            raise ValueError(
                "every feature is all-zero in at least one batch, so nothing can be "
                "harmonized. Filter low-count features before calling GLMComBat."
            )

        kept = values[keep]
        library = kept.sum(axis=0)
        phi_batch = np.zeros((kept.shape[0], n_batch))
        for b in range(n_batch):
            cols = np.flatnonzero(codes == b)
            sub = kept[:, cols]
            sub_offset = np.broadcast_to(np.log(sub.sum(axis=0)), sub.shape)
            sub_design, has_df = self._batch_design(within, cols, n_nonbatch, b)
            common = common_dispersion(sub, sub_design, sub_offset)
            phi_batch[:, b] = (
                featurewise_dispersion(sub, sub_design, common, sub_offset) if has_df else common
            )

        dispersion = phi_batch[:, codes]
        offset = np.broadcast_to(np.log(library), kept.shape)
        baseline = nb_glm_fit(kept, design, dispersion, offset, prior_count=_PRIOR_COUNT)

        alpha = baseline.coefficients[:, :n_batch] @ (sizes / n_samples)
        adjusted_offset = np.log(library)[None, :] + alpha[:, None]
        fit = nb_glm_fit(kept, design, dispersion, adjusted_offset, prior_count=_PRIOR_COUNT)

        self._alpha = alpha
        self._coefficients = fit.unshrunk_coefficients
        self._gamma = fit.coefficients[:, :n_batch]
        self._phi_batch = phi_batch
        self._phi_star = phi_batch.mean(axis=1)

    def _fit_met(
        self,
        values: FloatArray,
        codes: npt.NDArray[np.intp],
        design: FloatArray,
        within: FloatArray,
        sizes: FloatArray,
    ) -> None:
        n_batch = len(self._batch_levels)
        n_samples = values.shape[1]
        n_nonbatch = design.shape[1] - n_batch

        # Screen each batch with the design actually fitted, not the full `within`.
        adjustable = _has_residual_signal(values, design)
        for b in range(n_batch):
            cols = np.flatnonzero(codes == b)
            sub_design, _ = self._batch_design(within, cols, n_nonbatch, b)
            adjustable &= _has_residual_signal(values[:, cols], sub_design)
        self._adjustable = adjustable
        if not adjustable.any():
            raise ValueError(
                "no feature has residual variation left after the design is accounted "
                "for, so nothing can be harmonized. Check for constant features."
            )

        kept = values[adjustable]
        phi_batch = np.zeros((kept.shape[0], n_batch))
        for b in range(n_batch):
            cols = np.flatnonzero(codes == b)
            sub = kept[:, cols]
            sub_design, _ = self._batch_design(within, cols, n_nonbatch, b)
            common = common_precision(sub, sub_design)
            phi_batch[:, b] = featurewise_precision(sub, sub_design, common)

        fit = beta_glm_fit(kept, design, phi_batch[:, codes])
        batch_coef = fit.coefficients[:, :n_batch]
        log_phi_batch = np.log(phi_batch)

        if self._reference_index is not None:
            alpha_x = batch_coef[:, self._reference_index]
            alpha_z = log_phi_batch[:, self._reference_index]
        else:
            alpha_x = batch_coef @ (sizes / n_samples)
            alpha_z = log_phi_batch @ (sizes / n_samples)

        self._coefficients = fit.coefficients
        self._gamma = batch_coef - alpha_x[:, None]
        self._log_phi_batch = log_phi_batch
        self._alpha_z = alpha_z

    def transform(
        self,
        X: ArrayLike,
        *,
        batch: ArrayLike,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Harmonize samples using only the parameters frozen at fit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to harmonize. Columns must match those seen at fit.
        batch : array-like of shape (n_samples,)
            Batch labels; every level must have been seen at fit.
        discrete_covariates, continuous_covariates : array-like, optional
            The same covariates that were supplied at fit.

        Returns
        -------
        pd.DataFrame
            Harmonized data, same shape and index as ``X``.
        """
        if getattr(self, "_has_condition", False):
            raise NotImplementedError(
                "this model was fitted with a protected condition-of-interest term, which "
                "is unavailable at transform time. Such a configuration is whole-cohort "
                "only: call `fit_transform` on the complete cohort instead."
            )
        return self._apply(
            X,
            batch=batch,
            discrete_covariates=discrete_covariates,
            continuous_covariates=continuous_covariates,
            condition=None,
        )

    def _apply(
        self,
        X: ArrayLike,
        *,
        batch: ArrayLike,
        discrete_covariates: ArrayLike | None,
        continuous_covariates: ArrayLike | None,
        condition: ArrayLike | None,
    ) -> pd.DataFrame:
        """Shared body of ``transform`` and the whole-cohort ``fit_transform``."""
        if not hasattr(self, "_method"):
            raise ValueError("this GLMComBatModel is not fitted yet; call `fit` first.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._feature_names)
        if len(X.columns) != len(self._feature_names):
            raise ValueError(
                f"X has {len(X.columns)} features but the model was fitted on "
                f"{len(self._feature_names)}."
            )
        index = X.index

        batch = self._as_categorical(batch, index, "batch")
        unseen = set(batch.dropna().unique()) - set(self._batch_levels)
        if unseen:
            raise ValueError(
                f"batch level(s) {sorted(map(str, unseen))} were not seen at fit. The "
                f"inductive contract of GLMComBat is 'held-out samples from batches seen "
                f"at fit': a batch with no training samples has no estimated batch effect. "
                f"Refit including these samples, or harmonize that batch separately."
            )

        discrete = self._as_frame(discrete_covariates, index, "discrete_covariates")
        continuous = self._as_frame(continuous_covariates, index, "continuous_covariates")
        condition_frame = self._as_frame(condition, index, "condition")
        covariates = self._covariate_frame(
            index, discrete, continuous, condition_frame, fitting=False
        )
        expected = list(self._nonbatch_columns)
        if list(covariates.columns) != expected:
            missing = [c for c in expected if c not in covariates.columns]
            extra = [c for c in covariates.columns if c not in expected]
            if missing or extra:
                raise ValueError(
                    f"the covariate columns do not match those seen at fit "
                    f"(missing: {missing}, unexpected: {extra}). Pass the same covariates, "
                    f"with the same column labels, to transform as to fit."
                )
            covariates = covariates[expected]
        design = self._assemble(batch, covariates, self._method)

        values = self._check_values(X, self._method).T
        codes = np.asarray(
            pd.Categorical(batch, categories=self._batch_levels).codes, dtype=np.intp
        )

        if self._method == "seq":
            adjusted = self._transform_seq(values, codes, design)
        else:
            adjusted = self._transform_met(values, codes, design)

        out = pd.DataFrame(adjusted.T, index=index, columns=self._feature_names)
        if self._method == "seq":
            out = out.round(0)
            integral = {c: t for c, t in X.dtypes.items() if np.issubdtype(t, np.integer)}
            if integral:
                out = out.astype(integral)
        return out

    def _transform_seq(
        self, values: FloatArray, codes: npt.NDArray[np.intp], design: FloatArray
    ) -> FloatArray:
        out = values.copy()
        kept = values[self._adjustable]
        library = kept.sum(axis=0)
        if (library <= 0).any():
            raise ValueError(
                "at least one sample has a total count of zero over the features retained "
                "at fit, so its library size is undefined."
            )

        log_mu = np.log(library)[None, :] + self._alpha[:, None] + self._coefficients @ design.T
        mu_hat = np.exp(log_mu)
        mu_star = np.exp(log_mu - self._gamma[:, codes])

        adjusted = kept.copy()
        for b in range(len(self._batch_levels)):
            if b == self._reference_index:
                continue
            cols = np.flatnonzero(codes == b)
            if not len(cols):
                continue
            adjusted[:, cols] = map_to_batch_free_nb(
                kept[:, cols],
                mu_hat[:, cols],
                self._phi_batch[:, b],
                mu_star[:, cols],
                self._phi_star,
            )
        out[self._adjustable] = adjusted
        return out

    def _transform_met(
        self, values: FloatArray, codes: npt.NDArray[np.intp], design: FloatArray
    ) -> FloatArray:
        out = values.copy()
        kept = values[self._adjustable]

        logit_mu = self._coefficients @ design.T
        mu_hat = _expit(logit_mu)
        mu_star = _expit(logit_mu - self._gamma[:, codes])
        old_phi = np.exp(self._log_phi_batch[:, codes])
        new_phi = np.broadcast_to(np.exp(self._alpha_z)[:, None], kept.shape)

        out[self._adjustable] = map_to_batch_free_beta(kept, mu_hat, old_phi, mu_star, new_phi)
        return out

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        batch: ArrayLike,
        discrete_covariates: ArrayLike | None = None,
        continuous_covariates: ArrayLike | None = None,
        condition: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Fit on the whole cohort and harmonize it in one pass."""
        self.fit(
            X,
            batch=batch,
            discrete_covariates=discrete_covariates,
            continuous_covariates=continuous_covariates,
            condition=condition,
        )
        return self._apply(
            X,
            batch=batch,
            discrete_covariates=discrete_covariates,
            continuous_covariates=continuous_covariates,
            condition=condition,
        )

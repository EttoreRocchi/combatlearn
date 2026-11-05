"""Tests for GLMComBat ('seq' and 'met') and its GLM engines.

Covers reference parity, the per-observation inductive contract and the output domains.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from combatlearn import GLMComBat
from combatlearn.glm._interpolate import maximize_interpolant
from combatlearn.glm._model import GLMComBatModel
from combatlearn.glm._nb import adjusted_profile_likelihood, common_dispersion, nb_glm_fit

DATA = Path(__file__).parent / "data"


def simulate_counts(n_samples=30, n_features=40, n_batches=3, seed=0, with_covariate=False):
    """RNA-seq-like counts with a multiplicative per-feature batch effect."""
    rng = np.random.default_rng(seed)
    labels = np.array(list("ABCDEFG"))[:n_batches]
    batch = pd.Series(np.repeat(labels, n_samples // n_batches), name="batch")
    codes = pd.Categorical(batch).codes
    mu = (
        rng.gamma(5, 60, size=(n_features, 1))
        * np.exp(rng.normal(0, 0.4, size=(n_features, n_batches)))[:, codes]
    )
    covariate = None
    if with_covariate:
        covariate = pd.Series(np.tile(["x", "y"], len(batch) // 2), name="cov")
        mu = (
            mu
            * np.exp(rng.normal(0, 0.3, size=(n_features, 2)))[:, pd.Categorical(covariate).codes]
        )
    counts = rng.negative_binomial(5, 5 / (5 + mu)).astype(np.int64)
    return pd.DataFrame(counts.T), batch, covariate


def stratified_split(X, batch, n_held_out=4):
    """Hold out the last ``n_held_out`` samples of every batch, so all batches are seen at fit."""
    held_out = np.concatenate(
        [np.flatnonzero(np.asarray(batch) == level)[-n_held_out:] for level in batch.unique()]
    )
    mask = np.zeros(len(X), dtype=bool)
    mask[held_out] = True
    return X.loc[~mask], X.loc[mask], batch[~mask], batch[mask]


def simulate_betavalues(n_samples=24, n_features=30, seed=1):
    """Methylation-like beta-values with a per-feature batch shift on the logit scale."""
    rng = np.random.default_rng(seed)
    batch = pd.Series(np.repeat(["A", "B", "C"], n_samples // 3), name="batch")
    codes = pd.Categorical(batch).codes
    logit_mu = (
        rng.normal(0, 1.0, size=(n_features, 1))
        + rng.normal(0, 0.6, size=(n_features, 3))[:, codes]
    )
    mu = 1.0 / (1.0 + np.exp(-logit_mu))
    precision = np.exp(rng.normal(np.log(30), 0.3, size=(n_features, 1)))
    values = rng.beta(mu * precision, (1.0 - mu) * precision)
    return pd.DataFrame(np.clip(values, 1e-4, 1 - 1e-4).T), batch


@pytest.mark.parametrize("method", ["seq", "met"])
def test_transform_is_per_observation(method):
    """A held-out sample is transformed identically alone and in company."""
    if method == "seq":
        X, batch, _ = simulate_counts(n_samples=36, seed=3)
    else:
        X, batch = simulate_betavalues(n_samples=36, seed=3)
    train, test, batch_train, batch_test = stratified_split(X, batch)

    model = GLMComBatModel(method=method).fit(train, batch=batch_train)
    together = model.transform(test, batch=batch_test)
    one_at_a_time = pd.concat(
        [model.transform(test.iloc[[i]], batch=batch_test.iloc[[i]]) for i in range(len(test))]
    )
    pd.testing.assert_frame_equal(together, one_at_a_time)


@pytest.mark.parametrize("reference_batch", [None, "A"])
def test_transform_is_per_observation_with_covariates(reference_batch):
    """Per-observation transform also holds with a discrete covariate in the design."""
    X, batch, covariate = simulate_counts(n_samples=36, seed=4, with_covariate=True)
    train, test, batch_train, batch_test = stratified_split(X, batch)
    train_mask = X.index.isin(train.index)

    model = GLMComBatModel(method="seq", reference_batch=reference_batch).fit(
        train, batch=batch_train, discrete_covariates=covariate[train_mask]
    )
    kwargs = {"batch": batch_test, "discrete_covariates": covariate[~train_mask]}
    together = model.transform(test, **kwargs)
    one_at_a_time = pd.concat(
        [
            model.transform(
                test.iloc[[i]],
                batch=batch_test.iloc[[i]],
                discrete_covariates=covariate[~train_mask].iloc[[i]],
            )
            for i in range(len(test))
        ]
    )
    pd.testing.assert_frame_equal(together, one_at_a_time)


def test_covariate_encoding_is_frozen_at_fit():
    """Dummy coding uses the fit-time levels, not those present at transform."""
    X, batch, covariate = simulate_counts(n_samples=36, seed=7, with_covariate=True)
    model = GLMComBatModel(method="seq").fit(X, batch=batch, discrete_covariates=covariate)

    single_level = np.asarray(covariate) == "y"
    from_subset = model.transform(
        X[single_level], batch=batch[single_level], discrete_covariates=covariate[single_level]
    )
    from_full = model.transform(X, batch=batch, discrete_covariates=covariate)[single_level]
    pd.testing.assert_frame_equal(from_subset, from_full)


def test_unknown_covariate_column_at_transform_raises():
    """A covariate column absent at fit has no frozen levels to encode against."""
    X, batch, covariate = simulate_counts(seed=9, with_covariate=True)
    fitted = covariate.to_frame(name="cov")
    model = GLMComBatModel(method="seq").fit(X, batch=batch, discrete_covariates=fitted)
    with pytest.raises(ValueError, match="was not present at fit"):
        model.transform(X, batch=batch, discrete_covariates=fitted.rename(columns={"cov": "other"}))


def test_transform_ignores_the_composition_of_the_transform_set():
    """Adding unrelated samples to a transform call cannot change an existing row."""
    X, batch, _ = simulate_counts(n_samples=36, seed=8)
    train, test, batch_train, batch_test = stratified_split(X, batch)
    model = GLMComBatModel(method="seq").fit(train, batch=batch_train)

    from_subset = model.transform(test.iloc[:2], batch=batch_test.iloc[:2])
    from_full = model.transform(test, batch=batch_test).iloc[:2]
    pd.testing.assert_frame_equal(from_subset, from_full)


@pytest.mark.parametrize("method", ["seq", "met"])
def test_fit_transform_equals_fit_then_transform(method):
    """The whole-cohort shortcut agrees with the two-step path exactly."""
    if method == "seq":
        X, batch, _ = simulate_counts(seed=5)
    else:
        X, batch = simulate_betavalues(seed=5)
    est = GLMComBat(batch=batch, method=method)
    combined = est.fit_transform(X)
    separate = GLMComBat(batch=batch, method=method).fit(X).transform(X)
    pd.testing.assert_frame_equal(combined, separate)


def test_unseen_batch_at_transform_raises():
    """A batch not seen at fit raises at transform."""
    X, batch, _ = simulate_counts(n_samples=36, seed=6)
    train, test, batch_train, batch_test = stratified_split(X, batch)
    model = GLMComBatModel(method="seq").fit(train, batch=batch_train)
    with pytest.raises(ValueError, match="not seen at fit"):
        model.transform(test, batch=batch_test.map(lambda _: "Z"))


@pytest.mark.parametrize("with_covariate", [False, True])
@pytest.mark.parametrize("reference_batch", [None, "B"])
def test_seq_matches_inmoose(with_covariate, reference_batch):
    """fit_transform matches InMoose pycombat_seq exactly (skipped when InMoose is absent)."""
    pycombat_seq = pytest.importorskip("inmoose.pycombat", exc_type=ImportError).pycombat_seq

    X, batch, covariate = simulate_counts(seed=11, with_covariate=with_covariate)
    counts = X.values.T
    covar_mod = None if covariate is None else pd.DataFrame({"cov": covariate})
    expected = np.asarray(
        pycombat_seq(
            counts.copy(), np.asarray(batch), covar_mod=covar_mod, ref_batch=reference_batch
        )
    )
    obtained = GLMComBat(
        batch=batch,
        method="seq",
        discrete_covariates=covariate,
        reference_batch=reference_batch,
    ).fit_transform(X)
    np.testing.assert_array_equal(obtained.values.T, expected)


@pytest.mark.parametrize(
    ("fixture", "kwargs", "condition"),
    [("plain", {}, False), ("refbatch", {"reference_batch": "B"}, False), ("condition", {}, True)],
)
def test_met_matches_r_reference(fixture, kwargs, condition):
    """fit_transform matches the ComBat-met R package on committed fixtures.

    Fixtures come from ``tests/data/generate_met_fixtures.R``; R is not a test dependency.
    The tolerance absorbs R ``optimize`` vs SciPy ``minimize_scalar`` in the precision search.
    """
    values = pd.read_csv(DATA / "met_input.csv").values.T
    meta = pd.read_csv(DATA / "met_meta.csv")
    expected = pd.read_csv(DATA / f"met_expected_{fixture}.csv").values

    model = GLMComBatModel(method="met", **kwargs)
    obtained = model.fit_transform(
        pd.DataFrame(values),
        batch=meta["batch"],
        condition=meta["group"] if condition else None,
    )
    np.testing.assert_allclose(obtained.values.T, expected, rtol=1e-6, atol=1e-8)


def test_seq_preserves_integrality_and_dtype():
    """'seq' output is non-negative int64 counts."""
    X, batch, _ = simulate_counts(seed=12)
    out = GLMComBat(batch=batch, method="seq").fit_transform(X)
    assert (out.dtypes == np.int64).all()
    assert (out.values >= 0).all()


def test_seq_accepts_float_valued_counts():
    """Whole-number floats are accepted, and the output stays whole-numbered."""
    X, batch, _ = simulate_counts(seed=12)
    out = GLMComBat(batch=batch, method="seq").fit_transform(X.astype(float))
    assert np.array_equal(out.values, np.rint(out.values))


def test_met_output_stays_inside_the_unit_interval():
    """'met' output stays strictly inside (0, 1)."""
    X, batch = simulate_betavalues(seed=13)
    out = GLMComBat(batch=batch, method="met").fit_transform(X)
    assert ((out.values > 0.0) & (out.values < 1.0)).all()


def test_met_accepts_exact_zeros_and_ones():
    """Exact 0 and 1 are replaced by the pseudo-beta rather than rejected."""
    X, batch = simulate_betavalues(seed=14)
    X.iloc[0, 0] = 0.0
    X.iloc[1, 1] = 1.0
    out = GLMComBat(batch=batch, method="met").fit_transform(X)
    assert np.isfinite(out.values).all()


def test_zero_count_features_pass_through_unchanged():
    """Features all-zero in batch A are not adjusted and pass through unchanged."""
    X, batch, _ = simulate_counts(n_samples=36, seed=15)
    X.iloc[:12, :3] = 0
    train, test, _, _ = stratified_split(X, batch)
    est = GLMComBat(batch=batch, method="seq").fit(train)
    assert not est.adjustable_features_[:3].any()
    assert est.adjustable_features_[3:].all()

    out = est.transform(test)
    np.testing.assert_array_equal(out.values[:, :3], test.values[:, :3])


def test_single_sample_batch_is_rejected():
    """A single-sample batch is rejected at fit."""
    X, batch, _ = simulate_counts(seed=16)
    batch = batch.copy()
    batch.iloc[0] = "Z"
    with pytest.raises(ValueError, match="fewer than 2 samples"):
        GLMComBat(batch=batch, method="seq").fit(X)


def test_single_batch_is_rejected():
    """A single batch level is rejected at fit."""
    X, _, _ = simulate_counts(seed=17)
    with pytest.raises(ValueError, match="at least 2 are needed"):
        GLMComBat(batch=pd.Series(["A"] * len(X)), method="seq").fit(X)


def test_reference_batch_samples_are_untouched():
    """'seq' leaves reference-batch samples unchanged."""
    X, batch, _ = simulate_counts(seed=18)
    out = GLMComBat(batch=batch, method="seq", reference_batch="B").fit_transform(X)
    mask = np.asarray(batch == "B")
    np.testing.assert_array_equal(out.values[mask], X.values[mask])


def test_met_reference_batch_samples_are_untouched():
    """'met' leaves reference-batch samples unchanged up to the beta CDF/PPF round trip."""
    X, batch = simulate_betavalues(seed=18)
    out = GLMComBat(batch=batch, method="met", reference_batch="B").fit_transform(X)
    mask = np.asarray(batch == "B")
    np.testing.assert_allclose(out.values[mask], X.values[mask], rtol=0, atol=1e-12)


@pytest.mark.parametrize("method", ["seq", "met"])
def test_small_batch_with_a_covariate_is_still_adjustable(method):
    """A batch too small for the within-batch design still leaves features adjustable."""
    X, batch, covariate = simulate_counts(n_samples=30, seed=25, with_covariate=True)
    if method == "met":
        X = X.div(X.values.max() + 1.0)
    batch = batch.copy()
    batch.iloc[:2] = "Z"

    with pytest.warns(UserWarning, match="too few samples to support the within-batch design"):
        est = GLMComBat(
            batch=batch, method=method, discrete_covariates=covariate, min_samples_per_batch=2
        ).fit(X)
    assert est.adjustable_features_.any()


def test_seq_target_dispersion_is_the_more_cv_stable_choice():
    """Pinning the 'seq' target dispersion to the reference batch increases across-fold spread.

    Folds keep every batch but swing batch proportions; pinning is emulated via ``_phi_star``.
    """
    rng = np.random.default_rng(4)
    n_per, n_features, n_batches = 26, 40, 3
    batch = pd.Series(np.repeat(list("ABC"), n_per))
    codes = pd.Categorical(batch).codes
    mu = (
        rng.gamma(5, 60, size=(n_features, 1))
        * np.exp(rng.normal(0, 0.4, size=(n_features, n_batches)))[:, codes]
    )
    X = pd.DataFrame(rng.negative_binomial(5, 5 / (5 + mu)).astype(np.int64).T)

    held = np.concatenate([np.flatnonzero(codes == b)[-4:] for b in range(n_batches)])
    pool = np.setdiff1d(np.arange(len(X)), held)
    X_held, batch_held = (
        X.iloc[held].reset_index(drop=True),
        batch.iloc[held].reset_index(drop=True),
    )

    folds = []
    for r in range(5):
        g = np.random.default_rng(50 + r)
        keep = []
        for b in range(n_batches):
            sub = np.intersect1d(pool, np.flatnonzero(codes == b))
            keep.append(g.choice(sub, size=int((0.4 + 0.6 * g.random()) * len(sub)), replace=False))
        folds.append(np.sort(np.concatenate(keep)))

    def across_fold_spread(pin_to_reference):
        outputs = []
        for f in folds:
            est = GLMComBat(
                batch=batch.iloc[f].reset_index(drop=True), method="seq", reference_batch="A"
            ).fit(X.iloc[f].reset_index(drop=True))
            if pin_to_reference:
                model = est._model
                model._phi_star = model._phi_batch[:, model._reference_index].copy()
            est.batch = batch_held
            outputs.append(est.transform(X_held).values.astype(float))
        stacked = np.array(outputs)
        return np.median(stacked.std(0) / np.maximum(stacked.mean(0), 1.0))

    assert across_fold_spread(pin_to_reference=False) < across_fold_spread(pin_to_reference=True)


def test_confounded_covariate_is_rejected():
    """A covariate confounded with batch is rejected at fit."""
    X, batch, _ = simulate_counts(seed=19)
    confounded = pd.Series(np.where(np.asarray(batch) == "A", "u", "v"), name="cov")
    with pytest.raises(ValueError, match="confounded with batch"):
        GLMComBat(batch=batch, method="seq", discrete_covariates=confounded).fit(X)


def test_seq_rejects_non_integer_input():
    X, batch = simulate_betavalues(seed=20)
    with pytest.raises(ValueError, match="integer counts"):
        GLMComBat(batch=batch, method="seq").fit(X)


def test_seq_rejects_negative_input():
    X, batch, _ = simulate_counts(seed=21)
    X.iloc[0, 0] = -1
    with pytest.raises(ValueError, match="negative values"):
        GLMComBat(batch=batch, method="seq").fit(X)


def test_met_rejects_out_of_range_input():
    X, batch, _ = simulate_counts(seed=22)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        GLMComBat(batch=batch, method="met").fit(X)


@pytest.mark.parametrize("method", ["seq", "met"])
def test_null_batch_label_is_rejected(method):
    """A NaN batch label is rejected at fit."""
    X, batch, _ = simulate_counts(seed=26)
    if method == "met":
        X = X.div(X.values.max() + 1.0)
    batch = batch.astype(object).copy()
    batch.iloc[0] = np.nan
    with pytest.raises(ValueError, match=r"`batch` contains 1 NaN"):
        GLMComBat(batch=batch, method=method).fit(X)


def test_null_covariate_is_rejected():
    """A NaN discrete covariate is rejected at fit."""
    X, batch, covariate = simulate_counts(seed=27, with_covariate=True)
    covariate = covariate.astype(object).copy()
    covariate.iloc[0] = np.nan
    with pytest.raises(ValueError, match=r"`discrete_covariates` contains 1 NaN"):
        GLMComBat(batch=batch, method="seq", discrete_covariates=covariate).fit(X)


def test_relabelled_covariate_columns_at_transform_raise():
    """Covariate columns relabelled after fit raise at transform."""
    X, batch, _ = simulate_counts(seed=28)
    rng = np.random.default_rng(28)
    age = pd.DataFrame({"age": rng.normal(50, 10, size=len(X))})
    est = GLMComBat(batch=batch, method="seq", continuous_covariates=age).fit(X)

    est.continuous_covariates = pd.DataFrame(age.values)
    with pytest.raises(ValueError, match="do not match those seen at fit"):
        est.transform(X)


def test_unseen_covariate_level_at_transform_raises():
    """A covariate level not seen at fit raises at transform."""
    X, batch, covariate = simulate_counts(seed=29, with_covariate=True)
    est = GLMComBat(batch=batch, method="seq", discrete_covariates=covariate).fit(X)
    est.discrete_covariates = pd.Series(["never_seen"] * len(X), name="cov")
    with pytest.raises(ValueError, match=r"level\(s\) \['never_seen'\] that were not seen"):
        est.transform(X)


def test_transform_rejects_a_covariate_level_of_another_type():
    """A 0/1 covariate at transform is not the True/False level seen at fit."""
    X, batch, _ = simulate_counts(seed=29)
    flag = pd.Series(np.tile([True, False], len(X) // 2), index=X.index, name="flag")
    est = GLMComBat(batch=batch, method="seq", discrete_covariates=flag).fit(X)
    est.discrete_covariates = flag.astype(int)
    with pytest.raises(ValueError, match="were not seen at fit"):
        est.transform(X)


def test_degenerate_within_batch_design_warns():
    """A batch too small for the within-batch design warns about the fallback."""
    X, batch, covariate = simulate_counts(n_samples=30, seed=30, with_covariate=True)
    batch = batch.astype(object).copy()
    batch.iloc[:2] = "Z"
    with pytest.warns(UserWarning, match=r"too few samples to support the within-batch design"):
        GLMComBat(batch=batch, method="seq", discrete_covariates=covariate).fit(X)


def test_no_degenerate_warning_without_covariates():
    """A small batch without covariates does not warn."""
    X, batch, _ = simulate_counts(n_samples=30, seed=30)
    batch = batch.astype(object).copy()
    batch.iloc[:2] = "Z"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        GLMComBat(batch=batch, method="seq").fit(X)


def test_unknown_method_is_rejected():
    X, batch, _ = simulate_counts(seed=23)
    with pytest.raises(ValueError, match="is not recognized"):
        GLMComBat(batch=batch, method="biseq").fit(X)


def test_unfitted_transform_raises():
    X, batch, _ = simulate_counts(seed=24)
    with pytest.raises(Exception, match="not fitted"):
        GLMComBat(batch=batch, method="seq").transform(X)


def test_clone_and_pipeline():
    """The estimator clones and works as a Pipeline step."""
    X, batch, _ = simulate_counts(seed=25)
    est = GLMComBat(batch=batch, method="seq", reference_batch="A")
    cloned = clone(est)
    assert cloned.get_params()["reference_batch"] == "A"
    pipe = Pipeline([("harmonize", cloned)])
    assert pipe.fit_transform(X).shape == X.shape


def test_feature_names_out():
    X, batch, _ = simulate_counts(seed=26)
    X.columns = [f"gene{i}" for i in range(X.shape[1])]
    est = GLMComBat(batch=batch, method="seq").fit(X)
    np.testing.assert_array_equal(est.get_feature_names_out(), np.asarray(X.columns, dtype=object))


def test_numpy_input_is_accepted():
    X, batch, _ = simulate_counts(seed=27)
    out = GLMComBat(batch=np.asarray(batch), method="seq").fit_transform(X.values)
    assert out.shape == X.shape


def test_spline_maximizer_beats_every_grid_point():
    """The located maximum is interior and improves on the best grid value.

    Uses cosh, not a quadratic, whose near-zero cubic term would divide by rounding error.
    """
    grid = np.linspace(-6.0, 6.0, 11)
    peaks = np.array([-2.3, 0.4, 1.7])
    values = -np.cosh((grid[None, :] - peaks[:, None]) / 2.0)
    located = maximize_interpolant(grid, values)

    np.testing.assert_allclose(located, peaks, atol=0.2)
    interpolated = -np.cosh((located - peaks) / 2.0)
    assert (interpolated >= values.max(axis=1)).all()


def test_nb_glm_recovers_known_coefficients():
    """The NB fitter recovers the generating log-means on near-noiseless data."""
    rng = np.random.default_rng(30)
    design = np.repeat(np.eye(3), 40, axis=0)
    truth = rng.normal(np.log(500), 0.3, size=(20, 3))
    mu = np.exp(truth @ design.T)
    counts = rng.poisson(mu).astype(float)
    offset = np.zeros_like(counts)
    dispersion = np.full(counts.shape, 1e-6)
    fit = nb_glm_fit(counts, design, dispersion, offset)
    np.testing.assert_allclose(fit.coefficients, truth, atol=0.05)


def test_cox_reid_dispersion_recovers_the_generating_value():
    """The Cox-Reid common dispersion lands near the simulated one."""
    rng = np.random.default_rng(31)
    n_features, n_samples, true_phi = 300, 20, 0.25
    mu = np.full((n_features, n_samples), 400.0)
    size = 1.0 / true_phi
    counts = rng.negative_binomial(size, size / (size + mu)).astype(float)
    offset = np.zeros_like(counts)
    estimate = common_dispersion(counts, np.ones((n_samples, 1)), offset)
    assert 0.8 * true_phi < estimate < 1.25 * true_phi


def test_adjusted_profile_likelihood_peaks_at_the_true_dispersion():
    """The APL is maximized near the generating dispersion, not at the boundary."""
    rng = np.random.default_rng(32)
    true_phi = 0.2
    size = 1.0 / true_phi
    mu = np.full((150, 16), 300.0)
    counts = rng.negative_binomial(size, size / (size + mu)).astype(float)
    offset = np.zeros_like(counts)
    design = np.ones((16, 1))
    grid = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    totals = [
        float(np.sum(np.asarray(adjusted_profile_likelihood(p, counts, design, offset))))
        for p in grid
    ]
    assert int(np.argmax(totals)) == 2

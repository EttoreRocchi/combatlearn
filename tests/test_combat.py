import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import (
    residual_batch_gap,
    simulate_covariate_data,
    simulate_data,
    simulate_species_covbat_data,
    within_batch_species_gap,
)

from combatlearn import ComBat
from combatlearn.core import ComBatModel

DATA = Path(__file__).parent / "data"


_R_REFERENCE_CASES = [
    ("johnson_parametric", {"method": "johnson"}, 1e-6),
    ("johnson_nonparametric", {"method": "johnson", "parametric": False}, 1e-10),
    ("fortin_parametric", {"method": "fortin"}, 1e-6),
    ("fortin_nonparametric", {"method": "fortin", "parametric": False}, 1e-10),
    ("fortin_meanonly", {"method": "fortin", "mean_only": True}, 1e-6),
    ("chen_parametric", {"method": "chen"}, 1e-6),
    ("chen_nonparametric", {"method": "chen", "parametric": False}, 1e-10),
    ("chen_meanonly", {"method": "chen", "mean_only": True}, 1e-6),
]


@pytest.mark.parametrize(
    ("fixture", "kwargs", "tol"), _R_REFERENCE_CASES, ids=[c[0] for c in _R_REFERENCE_CASES]
)
def test_matches_r_reference(fixture, kwargs, tol):
    """fit_transform reproduces the reference R ComBat and CovBat on committed fixtures.

    Fixtures come from ``tests/data/generate_combat_fixtures.R``; R is not a test dependency.
    ``eps=1e-14`` because R adds no variance floor; the 1e-6 cases differ only by the
    stopping rule of the parametric EB iteration.
    """
    values = pd.read_csv(DATA / "combat_input.csv").values.T
    meta = pd.read_csv(DATA / "combat_meta.csv")
    expected = pd.read_csv(DATA / f"combat_expected_{fixture}.csv").values

    kwargs = {"batch": meta["batch"], "eps": 1e-14, **kwargs}
    if kwargs["method"] != "johnson":
        kwargs["discrete_covariates"] = meta[["group"]]
    if kwargs["method"] == "chen":
        kwargs["covbat_cov_thresh"] = 3
    obtained = ComBat(**kwargs).fit_transform(pd.DataFrame(values))
    np.testing.assert_allclose(obtained.values.T, expected, rtol=tol, atol=tol)


def test_transform_without_fit_raises():
    """Test that `transform` raises a `ValueError` if not fitted."""
    X, batch = simulate_data()
    model = ComBatModel()
    with pytest.raises(ValueError, match="not fitted"):
        model.transform(X, batch=batch)


def test_unseen_batch_raises_value_error():
    """Test that unseen batch raises a `ValueError`."""
    X, batch = simulate_data()
    model = ComBatModel().fit(X, batch=batch)
    new_batch = pd.Series(["Z"] * len(batch), index=batch.index)
    with pytest.raises(ValueError):
        model.transform(X, batch=new_batch)


def test_single_sample_batch_error():
    """Test that a single sample batch raises a `ValueError`."""
    X, batch = simulate_data()
    batch.iloc[0] = "single"
    with pytest.raises(ValueError):
        ComBatModel().fit(X, batch=batch)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_dtypes_preserved(method):
    """All output columns must remain floating dtypes after correction."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    X_corr = ComBat(batch=batch, method=method, **extra).fit_transform(X)
    assert all(np.issubdtype(dt, np.floating) for dt in X_corr.dtypes)


def test_wrapper_clone_and_pipeline():
    """Test `ComBat` wrapper can be cloned and used in a `Pipeline`."""
    X, batch = simulate_data()
    wrapper = ComBat(batch=batch, parametric=True)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("combat", wrapper),
        ]
    )
    X_corr = pipe.fit_transform(X)
    pipe_clone: Pipeline = clone(pipe)
    X_corr2 = pipe_clone.fit_transform(X)
    np.testing.assert_allclose(X_corr, X_corr2, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_no_nan_or_inf_in_output(method):
    """`ComBat` must not introduce NaN or Inf values, for any backend."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    X_corr = ComBat(batch=batch, method=method, **extra).fit_transform(X)
    assert not np.isnan(X_corr.values).any()
    assert not np.isinf(X_corr.values).any()


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_shape_preserved(method):
    """The (n_samples, n_features) shape must be identical pre- and post-ComBat."""
    if method == "johnson":
        X, batch = simulate_data()
        combat = ComBat(batch=batch, method=method).fit(X)
    elif method in ["fortin", "chen"]:
        X, batch, disc, cont = simulate_covariate_data()
        combat = ComBat(
            batch=batch,
            discrete_covariates=disc,
            continuous_covariates=cont,
            method=method,
        ).fit(X)

    X_corr = combat.transform(X)
    assert X_corr.shape == X.shape


def test_johnson_print_warning():
    """Test that a warning is printed when using the Johnson method."""
    X, batch, disc, cont = simulate_covariate_data()
    with pytest.warns(Warning, match="Covariates are ignored when using method='johnson'."):
        _ = ComBat(
            batch=batch,
            discrete_covariates=disc,
            continuous_covariates=cont,
            method="johnson",
        ).fit(X)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_reference_batch_samples_unchanged(method):
    """Samples belonging to the reference batch must come out numerically identical."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    elif method in ["fortin", "chen"]:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    ref_batch = batch.iloc[0]
    combat = ComBat(batch=batch, method=method, reference_batch=ref_batch, **extra).fit(X)
    X_corr = combat.transform(X)

    mask = batch == ref_batch
    np.testing.assert_allclose(X_corr.loc[mask].values, X.loc[mask].values, rtol=0, atol=1e-10)


def test_reference_batch_missing_raises():
    """Asking for a reference batch that doesn't exist should fail."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="not found"):
        ComBat(batch=batch, reference_batch="DOES_NOT_EXIST").fit(X)


@pytest.mark.parametrize("parametric", [True, False])
@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_parametric_vs_nonparametric(parametric, method):
    """Test both parametric and non-parametric modes work without errors."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    combat = ComBat(batch=batch, method=method, parametric=parametric, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


@pytest.mark.parametrize("mean_only", [True, False])
@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_mean_only_mode(mean_only, method):
    """Test mean_only mode works for all methods."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    combat = ComBat(batch=batch, method=method, mean_only=mean_only, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_covbat_cov_thresh_as_float():
    """Test CovBat with covbat_cov_thresh as float."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method="chen",
        covbat_cov_thresh=0.95,
    )
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_covbat_cov_thresh_as_int():
    """Test CovBat with covbat_cov_thresh as int."""
    X, batch, disc, cont = simulate_covariate_data()
    n_components = 10
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method="chen",
        covbat_cov_thresh=n_components,
    )
    X_corr = combat.fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()
    assert combat._model._covbat_n_pc == n_components


@pytest.mark.parametrize("seed", [9, 10, 13])
def test_covbat_cov_thresh_one_keeps_every_component(seed):
    """covbat_cov_thresh=1.0 keeps every PC, even when the cumulative ratio rounds below 1."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((60, 8)))
    batch = pd.Series(np.repeat(["A", "B"], 30))
    combat = ComBat(batch=batch, method="chen", covbat_cov_thresh=1.0)
    out = combat.fit_transform(X)
    if np.cumsum(combat._model._covbat_pca.explained_variance_ratio_)[-1] >= 1.0:
        pytest.skip("the cumulative ratio reaches 1.0 here, so the rounding case does not arise")
    assert np.isfinite(out.values).all()
    assert combat._model._covbat_n_pc == X.shape[1]


@pytest.mark.parametrize("thresh", [1.0, 20])
def test_covbat_never_keeps_a_pc_without_training_variance(thresh):
    """With fewer training rows than features, CovBat skips the PCs with no variance.

    20 centered rows give 19 PCs with variance, whose ratios sum below 1.0 in floating point.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 50)))
    batch = pd.Series(np.tile(["A", "B"], 20))
    train = np.arange(40) < 20
    chen = ComBat(batch=batch, method="chen", covbat_cov_thresh=thresh).fit(X[train])
    fortin = ComBat(batch=batch, method="fortin").fit(X[train])

    n_with_variance = train.sum() - 1
    if isinstance(thresh, float):
        cumulative = np.cumsum(chen._model._covbat_pca.explained_variance_ratio_)
        assert cumulative[n_with_variance - 1] < thresh
    assert chen._model._covbat_n_pc <= n_with_variance
    change = chen.transform(X[~train]).to_numpy() - fortin.transform(X[~train]).to_numpy()
    assert np.abs(change).max() < 10, f"held-out change {np.abs(change).max():.1e}"


@pytest.mark.parametrize("parametric", [True, False], ids=["parametric", "nonparametric"])
def test_johnson_reference_batch_gets_the_reference_mean_and_variance(parametric):
    """Every other batch is moved onto the reference batch's mean and variance.

    The non-parametric estimate pools over features, hence its looser mean tolerance.
    """
    X, batch = simulate_data(n_samples=3000, n_features=25)
    X = X * np.where(batch.to_numpy()[:, None] == "B", 1.8, 1.0)
    out = ComBat(
        batch=batch, method="johnson", reference_batch="A", parametric=parametric
    ).fit_transform(X)
    groups = out.groupby(batch.to_numpy())
    mean, sd = groups.mean(), groups.std()
    mean_tol = 0.05 if parametric else 0.15
    for lvl in ["B", "C"]:
        np.testing.assert_allclose(mean.loc[lvl], mean.loc["A"], atol=mean_tol)
        np.testing.assert_allclose(sd.loc[lvl], sd.loc["A"], rtol=0.05)


@pytest.mark.parametrize("parametric", [True, False], ids=["parametric", "nonparametric"])
def test_johnson_small_reference_batch_sets_the_target(parametric):
    """Other batches land on a small reference batch's own mean and variance."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(np.vstack([rng.normal(2.0, 1.0, (5, 30)), rng.normal(-1.0, 2.5, (200, 30))]))
    batch = pd.Series(["R"] * 5 + ["O"] * 200)
    with pytest.warns(UserWarning, match="highly imbalanced"):
        out = ComBat(
            batch=batch, method="johnson", reference_batch="R", parametric=parametric
        ).fit_transform(X)
    ref, other = out.to_numpy()[:5], out.to_numpy()[5:]
    np.testing.assert_allclose(other.var(0, ddof=1) / ref.var(0, ddof=1), 1.0, atol=0.1)
    gap = np.abs(other.mean(0) - ref.mean(0)) / ref.std(0, ddof=1)
    assert gap.max() < 0.3, f"mean gap {gap.max():.2f} reference SD"


def test_transform_rejects_a_covariate_level_of_another_type():
    """A 0/1 covariate at transform is not the True/False level seen at fit."""
    X, batch = simulate_data()
    flag = pd.Series(np.tile([True, False], len(X) // 2), index=X.index, name="flag")
    est = ComBat(batch=batch, discrete_covariates=flag.to_frame(), method="fortin").fit(X)
    est.discrete_covariates = flag.astype(int).to_frame()
    with pytest.raises(ValueError, match="were not seen at fit"):
        est.transform(X)


def test_transform_rejects_a_category_absent_from_the_training_rows():
    """A category-dtype level with no training rows counts as unseen at transform."""
    X, batch = simulate_data()
    sex = pd.Series(pd.Categorical(np.tile(["M", "F", "X"], 50)), index=X.index, name="sex")
    train = (sex != "X").to_numpy()
    est = ComBat(batch=batch, discrete_covariates=sex.to_frame(), method="fortin").fit(X[train])
    with pytest.raises(ValueError, match="were not seen at fit"):
        est.transform(X[~train])


def test_covbat_skips_pcs_with_negligible_training_variance():
    """A near-duplicate feature leaves a ~0-variance PC that must not be retained."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((40, 6))
    X = pd.DataFrame(np.column_stack([base, base[:, 0] + 1e-8 * rng.standard_normal(40)]))
    batch = pd.Series(np.tile(["A", "B"], 20))
    train = np.arange(40) < 30
    held = X[~train].copy()
    held.iloc[:, 6] = held.iloc[:, 0] + 0.1
    chen = ComBat(batch=batch, method="chen", covbat_cov_thresh=7).fit(X[train])
    fortin = ComBat(batch=batch, method="fortin").fit(X[train])

    assert chen._model._covbat_n_pc <= 6
    change = chen.transform(held).to_numpy() - fortin.transform(held).to_numpy()
    assert np.abs(change).max() < 10, f"held-out change {np.abs(change).max():.1e}"


def test_covbat_reference_batch_matches_the_reference_score_moments():
    """With reference_batch, CovBat moves every batch onto the reference batch's PC scores."""
    X, batch, disc = simulate_species_covbat_data()
    est = ComBat(batch=batch, discrete_covariates=disc, method="chen", reference_batch="A").fit(X)
    model = est._model
    _, stand_mu = model._fortin_residuals(X, batch.astype("category"), disc, None)
    resid = est.transform(X).to_numpy() - stand_mu
    scores = model._covbat_pca.transform(resid)[:, : model._covbat_n_pc]
    b = batch.to_numpy()
    for lvl in ["B", "C"]:
        np.testing.assert_allclose(scores[b == lvl].mean(0), scores[b == "A"].mean(0), atol=1e-6)
        np.testing.assert_allclose(
            scores[b == lvl].var(0, ddof=1), scores[b == "A"].var(0, ddof=1), rtol=1e-6
        )


def test_covbat_cov_thresh_invalid_float_raises():
    """Test that invalid float values for covbat_cov_thresh raise ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        ComBatModel(covbat_cov_thresh=1.5)
    with pytest.raises(ValueError, match="out of range"):
        ComBatModel(covbat_cov_thresh=0.0)


def test_covbat_cov_thresh_invalid_int_raises():
    """Test that invalid int values for covbat_cov_thresh raise ValueError."""
    with pytest.raises(ValueError, match="invalid"):
        ComBatModel(covbat_cov_thresh=0)


def test_covbat_cov_thresh_invalid_type_raises():
    """Test that invalid types for covbat_cov_thresh raise TypeError."""
    with pytest.raises(TypeError, match="must be float or int"):
        ComBatModel(covbat_cov_thresh="invalid")


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_index_preserved(method):
    """Test that the index is preserved after transformation."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    custom_index = pd.Index([f"sample_{i}" for i in range(len(X))])
    X.index = custom_index
    batch.index = custom_index
    if method != "johnson":
        disc.index = custom_index
        cont.index = custom_index

    combat = ComBat(batch=batch, method=method, **extra)
    X_corr = combat.fit_transform(X)
    assert X_corr.index.equals(custom_index)


@pytest.mark.parametrize("method", ["johnson", "fortin", "chen"])
def test_column_names_preserved(method):
    """Test that column names are preserved after transformation."""
    if method == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}

    custom_columns = [f"feature_{i}" for i in range(X.shape[1])]
    X.columns = custom_columns

    combat = ComBat(batch=batch, method=method, **extra)
    X_corr = combat.fit_transform(X)
    assert list(X_corr.columns) == custom_columns


def test_invalid_method_raises():
    """Test that an invalid method raises ValueError."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="not recognized"):
        ComBatModel(method="invalid").fit(X, batch=batch)


def test_fit_nan_in_X_raises():
    """Fitting with NaN in X must raise ValueError."""
    X, batch = simulate_data()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ComBat(batch=batch).fit(X)


def test_fit_inf_in_X_raises():
    """Fitting with Inf in X must raise ValueError."""
    X, batch = simulate_data()
    X.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        ComBat(batch=batch).fit(X)


def test_transform_nan_in_X_raises():
    """Transforming with NaN in X must raise ValueError."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch).fit(X)
    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        combat.transform(X_bad)


def test_fit_nan_in_batch_raises():
    """Fitting with NaN in batch must raise ValueError."""
    X, batch = simulate_data()
    batch.iloc[0] = np.nan
    with pytest.raises(ValueError, match=r"batch contains.*NaN"):
        ComBat(batch=batch).fit(X)


def test_fit_nan_in_discrete_covariates_raises():
    """Fitting with NaN in discrete covariates must raise ValueError."""
    X, batch, disc, cont = simulate_covariate_data()
    disc.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"discrete_covariates contains.*NaN"):
        ComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
        ).fit(X)


def test_fit_nan_in_continuous_covariates_raises():
    """Fitting with NaN in continuous covariates must raise ValueError."""
    X, batch, disc, cont = simulate_covariate_data()
    cont.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        ComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
        ).fit(X)


def test_get_feature_names_out_dataframe():
    """get_feature_names_out returns column names from DataFrame input."""
    X, batch = simulate_data(n_samples=100, n_features=10)
    X.columns = [f"feat_{i}" for i in range(10)]
    combat = ComBat(batch=batch).fit(X)
    names = combat.get_feature_names_out()
    assert list(names) == [f"feat_{i}" for i in range(10)]


def test_get_feature_names_out_ndarray():
    """get_feature_names_out returns default names for numpy input."""
    X, batch = simulate_data(n_samples=100, n_features=10)
    combat = ComBat(batch=batch).fit(X.values)
    names = combat.get_feature_names_out()
    assert list(names) == [f"x{i}" for i in range(10)]


def test_get_feature_names_out_not_fitted_raises():
    """get_feature_names_out before fit raises NotFittedError."""
    _X, batch = simulate_data()
    combat = ComBat(batch=batch)
    with pytest.raises(NotFittedError):
        combat.get_feature_names_out()


def test_set_output_pandas_in_pipeline():
    """Pipeline with set_output(transform='pandas') works with ComBat."""
    X, batch = simulate_data(n_samples=100, n_features=10)
    X.columns = [f"feat_{i}" for i in range(10)]
    pipe = Pipeline([("combat", ComBat(batch=batch))])
    pipe.set_output(transform="pandas")
    result = pipe.fit_transform(X)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [f"feat_{i}" for i in range(10)]


def test_set_params_updates_method():
    """set_params must propagate to the model created at fit time."""
    X, batch = simulate_data()
    combat = ComBat(batch=batch, method="johnson")
    combat.set_params(method="fortin")
    combat.fit(X)
    assert combat._model.method == "fortin"


@pytest.mark.parametrize("method", ["fortin", "chen"])
def test_transform_subset_of_batches(method):
    """Transform with a subset of fitted batches must not crash."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method=method,
    ).fit(X)

    mask = batch != "C"
    X_sub = X.loc[mask]
    X_corr = combat.transform(X_sub)
    assert X_corr.shape == X_sub.shape
    assert not np.isnan(X_corr.values).any()


@pytest.mark.parametrize("method", ["fortin", "chen"])
def test_transform_new_covariate_levels_raises(method):
    """An unseen discrete covariate level at transform raises a ValueError."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch,
        discrete_covariates=disc,
        continuous_covariates=cont,
        method=method,
    ).fit(X)

    disc_new = disc.copy()
    disc_new.iloc[0, 0] = "diag_3"
    combat_new = ComBat(
        batch=batch,
        discrete_covariates=disc_new,
        continuous_covariates=cont,
        method=method,
    )
    combat_new._model = combat._model
    combat_new.feature_names_in_ = combat.feature_names_in_
    with pytest.raises(ValueError, match=r"level\(s\) \['diag_3'\] that were not seen at fit"):
        combat_new.transform(X)


def test_transform_nan_in_batch_raises():
    """A NaN batch label at transform raises a ValueError."""
    X, batch, disc, cont = simulate_covariate_data()
    combat = ComBat(
        batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
    ).fit(X)
    bad = batch.astype(object).copy()
    bad.iloc[0] = np.nan
    combat.batch = bad
    with pytest.raises(ValueError, match=r"batch contains.*NaN"):
        combat.transform(X)


def _target_leak_frame(n=40, seed=0):
    """Data where one covariate is a relabeling of the target and one is innocent."""
    rng = np.random.default_rng(seed)
    batch = pd.Series(np.tile(["A", "B"], n // 2))
    y = pd.Series(np.repeat([0, 1], n // 2))
    relabeled = pd.Series(np.where(y == 0, "ctrl", "case"))
    sex = pd.Series(rng.choice(["m", "f"], size=n))
    return pd.DataFrame(rng.normal(size=(n, 10))), batch, y, relabeled, sex


def test_target_as_covariate_warns():
    """A covariate that determines y triggers the target-leak warning.

    The covariate holds the original labels while y is encoded, so equality would miss it.
    """
    X, batch, y, relabeled, _ = _target_leak_frame()
    with pytest.warns(UserWarning, match=r"determines `y` on the training data"):
        ComBat(batch=batch, method="fortin", discrete_covariates=relabeled).fit(X, y)


def _assert_no_target_leak_warning(fit_call):
    """Assert the target-leak check stays silent, ignoring unrelated (e.g. rank) warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_call()
    leaked = [str(w.message) for w in caught if "determines `y`" in str(w.message)]
    assert not leaked, leaked


def test_target_leak_check_is_quiet_on_a_legitimate_covariate():
    """A covariate unrelated to the target must not trip the check."""
    X, batch, y, _, sex = _target_leak_frame()
    _assert_no_target_leak_warning(
        lambda: ComBat(batch=batch, method="fortin", discrete_covariates=sex).fit(X, y)
    )


def test_target_leak_check_needs_y():
    """Without y there is nothing to compare against, so the check cannot fire."""
    X, batch, _, relabeled, _ = _target_leak_frame()
    _assert_no_target_leak_warning(
        lambda: ComBat(batch=batch, method="fortin", discrete_covariates=relabeled).fit(X)
    )


def test_target_leak_check_skips_near_unique_covariates():
    """The check skips covariates with fewer than two samples per level."""
    X, batch, y, _, _ = _target_leak_frame()
    ids = pd.Series([f"s{i}" for i in range(len(X))])
    _assert_no_target_leak_warning(
        lambda: ComBat(batch=batch, method="fortin", discrete_covariates=ids).fit(X, y)
    )


def test_target_leak_warning_fires_inside_a_cv_pipeline():
    """The target-leak warning surfaces from cross_val_score on a Pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    X, batch, y, relabeled, _ = _target_leak_frame()
    pipe = Pipeline(
        [
            ("cb", ComBat(batch=batch, method="fortin", discrete_covariates=relabeled)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    with pytest.warns(UserWarning, match=r"determines `y`"):
        cross_val_score(pipe, X, y, cv=StratifiedKFold(3))


def test_error_message_includes_received_value():
    """Error messages should include the received value."""
    with pytest.raises(ValueError, match=r"1\.5"):
        ComBatModel(covbat_cov_thresh=1.5)
    with pytest.raises(TypeError, match="str"):
        ComBatModel(covbat_cov_thresh="bad")


def test_error_message_batch_too_small_includes_count():
    """Batch-too-small error should include sample count and suggestion."""
    X, batch = simulate_data()
    batch.iloc[0] = "single"
    with pytest.raises(ValueError, match=r"1 sample.*Consider merging"):
        ComBatModel().fit(X, batch=batch)


def test_error_message_reference_batch_lists_available():
    """reference_batch error should list available batches."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="Available batches"):
        ComBatModel(reference_batch="MISSING").fit(X, batch=batch)


def test_near_zero_variance_warns():
    """Near-zero variance features should trigger a warning."""
    X, batch = simulate_data(n_samples=60, n_features=10)
    X["constant"] = 0.0
    with pytest.warns(UserWarning, match="near-zero variance"):
        ComBat(batch=batch).fit(X)


def test_imbalanced_batches_warns():
    """Highly imbalanced batches should trigger a warning."""
    rng = np.random.default_rng(42)
    n_large, n_small = 200, 5
    n = n_large + 2 * n_small
    X = pd.DataFrame(rng.standard_normal((n, 10)))
    batch = pd.Series(["A"] * n_large + ["B"] * n_small + ["C"] * n_small)
    with pytest.warns(UserWarning, match="imbalanced"):
        ComBat(batch=batch).fit(X)


def test_collinear_covariate_warns():
    """Covariate perfectly collinear with batch should trigger a warning."""
    X, batch, _disc, _cont = simulate_covariate_data()
    disc_collinear = batch.to_frame("cov")
    with pytest.warns(UserWarning, match="rank-deficient"):
        ComBat(
            batch=batch,
            discrete_covariates=disc_collinear,
            method="fortin",
        ).fit(X)


def test_numpy_x_shorter_than_batch_raises():
    """A numpy X shorter than the construction-time batch errors clearly (the CV footgun)."""
    X, batch = simulate_data(n_samples=60, n_features=10)
    cb = ComBat(batch=batch, method="johnson")
    X_fold = np.asarray(X)[:30]
    with pytest.raises(ValueError, match="positionally"):
        cb.fit(X_fold)


def test_numpy_x_and_batch_same_length_ok():
    """Numpy X with a same-length numpy batch aligns positionally and fits fine."""
    X, batch = simulate_data(n_samples=60, n_features=10)
    out = ComBat(batch=np.asarray(batch), method="johnson").fit_transform(np.asarray(X))
    assert out.shape == (60, 10)
    assert np.isfinite(out.values).all()


def test_cross_val_score_numpy_raises_clear_error():
    """cross_val_score over numpy X + numpy batch surfaces a clear length error, not a crash."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, batch = simulate_data(n_samples=60, n_features=10)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=60)
    pipe = Pipeline(
        [
            ("combat", ComBat(batch=np.asarray(batch), method="johnson")),
            ("clf", LogisticRegression()),
        ]
    )
    with pytest.raises(ValueError, match="positionally"):
        cross_val_score(pipe, np.asarray(X), y, cv=3, error_score="raise")


def test_balanced_batches_no_imbalance_warning():
    """Balanced batches should not trigger imbalance warning."""
    X, batch = simulate_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ComBat(batch=batch).fit(X)


@pytest.mark.parametrize(
    "alias,canonical",
    [("classic_combat", "johnson"), ("neurocombat", "fortin"), ("covbat", "chen")],
)
def test_method_alias_matches_canonical(alias, canonical):
    """A literature alias must produce the same result as its canonical author name."""
    if canonical == "johnson":
        X, batch = simulate_data()
        extra = {}
    else:
        X, batch, disc, cont = simulate_covariate_data()
        extra = {"discrete_covariates": disc, "continuous_covariates": cont}
    out_alias = ComBat(batch=batch, method=alias, **extra).fit_transform(X)
    out_canon = ComBat(batch=batch, method=canonical, **extra).fit_transform(X)
    np.testing.assert_allclose(out_alias.values, out_canon.values, rtol=1e-10, atol=1e-10)


def test_method_alias_case_and_separator_insensitive():
    """Alias matching ignores case, hyphens, underscores, and spaces."""
    X, batch, disc, cont = simulate_covariate_data()
    base = ComBat(
        batch=batch, discrete_covariates=disc, continuous_covariates=cont, method="fortin"
    ).fit_transform(X)
    for name in ["neuroCombat", "neuro-combat", "NEUROCOMBAT", "neuro_combat"]:
        out = ComBat(
            batch=batch, discrete_covariates=disc, continuous_covariates=cont, method=name
        ).fit_transform(X)
        np.testing.assert_allclose(out.values, base.values, rtol=1e-10, atol=1e-10)


def test_classic_combat_separator_variants():
    """'classic_combat', 'classic-combat', 'ClassicComBat' all resolve to johnson."""
    X, batch = simulate_data()
    base = ComBat(batch=batch, method="johnson").fit_transform(X)
    for name in ["classic_combat", "classic-combat", "ClassicComBat"]:
        out = ComBat(batch=batch, method=name).fit_transform(X)
        np.testing.assert_allclose(out.values, base.values, rtol=1e-10, atol=1e-10)


def test_invalid_method_message_lists_aliases():
    """The unrecognized-method error mentions the alias names."""
    X, batch = simulate_data()
    with pytest.raises(ValueError, match="classic_combat"):
        ComBat(batch=batch, method="not_a_method").fit(X)


def _covariate_cohort(n=60, n_features=8, seed=0):
    """A cohort whose discrete covariate carries a large, easily visible effect."""
    rng = np.random.default_rng(seed)
    batch = pd.Series(np.repeat(["A", "B", "C"], n // 3), name="batch")
    sex = pd.Series(np.tile(["M", "F"], n // 2), name="sex")
    values = rng.standard_normal((n, n_features)) + (np.asarray(sex) == "M")[:, None] * 3.0
    return pd.DataFrame(values), batch, sex


@pytest.mark.parametrize("method", ["fortin", "chen", "gam", "covbat_gam"])
def test_discrete_covariate_encoding_is_frozen_at_fit(method):
    """Transforming a sample must not depend on which other samples accompany it."""
    X, batch, sex = _covariate_cohort()
    kwargs = {"batch": batch, "method": method, "discrete_covariates": sex}
    if method in ("gam", "covbat_gam"):
        kwargs["continuous_covariates"] = pd.DataFrame({"age": np.linspace(20, 80, len(X))})
        kwargs["spline_df"] = 5

    combat = ComBat(**kwargs).fit(X)
    together = combat.transform(X)
    one_at_a_time = pd.concat([combat.transform(X.iloc[[i]]) for i in range(len(X))])
    np.testing.assert_allclose(together.values, one_at_a_time.values, rtol=1e-10, atol=1e-10)


def test_transform_of_a_single_covariate_level_matches_the_full_transform():
    """A subset holding only one covariate level is corrected like the full cohort."""
    X, batch, sex = _covariate_cohort()
    combat = ComBat(batch=batch, method="fortin", discrete_covariates=sex).fit(X)
    full = combat.transform(X)
    mask = np.asarray(sex) == "M"
    np.testing.assert_allclose(
        combat.transform(X[mask]).values, full[mask].values, rtol=1e-10, atol=1e-10
    )


def test_covbat_preserves_discrete_covariate_effect():
    """CovBat keeps the protected covariate effect that Fortin keeps and removes the batch gap.

    The species covariate is unbalanced across batches, so it correlates with batch.
    """
    X, batch, disc = simulate_species_covbat_data()
    kwargs = {"batch": batch, "discrete_covariates": disc}
    fortin = ComBat(method="fortin", **kwargs).fit_transform(X)
    chen = ComBat(method="chen", **kwargs).fit_transform(X)

    fortin_gap = within_batch_species_gap(fortin, batch, disc)
    chen_gap = within_batch_species_gap(chen, batch, disc)
    assert chen_gap >= 0.9 * fortin_gap, (
        f"CovBat species gap {chen_gap:.3f} < 0.9 x Fortin gap {fortin_gap:.3f}"
    )

    raw_batch_gap = residual_batch_gap(X, batch, disc)
    chen_batch_gap = residual_batch_gap(chen, batch, disc)
    assert chen_batch_gap <= 0.05 * raw_batch_gap, (
        f"CovBat residual batch gap {chen_batch_gap:.3f} > 5% of raw {raw_batch_gap:.3f}"
    )


def test_covbat_keeps_the_pooled_variance_of_the_pc_scores():
    """CovBat brings each batch's PC-score variance to the pooled one, not to 1.

    Checked on the total variance, which must stay within 10% of Fortin's.
    """
    X, batch, disc = simulate_species_covbat_data()
    kwargs = {"batch": batch, "discrete_covariates": disc}
    fortin = ComBat(method="fortin", **kwargs).fit_transform(X)
    chen = ComBat(method="chen", **kwargs).fit_transform(X)

    ratio = chen.var().sum() / fortin.var().sum()
    assert 0.9 <= ratio <= 1.1, f"CovBat / Fortin total variance ratio {ratio:.3f}"


def test_covbat_without_covariates_keeps_the_within_batch_signal():
    """Without covariates CovBat keeps the within-batch species gap that Fortin keeps."""
    X, batch, disc = simulate_species_covbat_data()
    fortin = ComBat(batch=batch, method="fortin").fit_transform(X)
    chen = ComBat(batch=batch, method="chen").fit_transform(X)

    fortin_gap = within_batch_species_gap(fortin, batch, disc)
    chen_gap = within_batch_species_gap(chen, batch, disc)
    assert chen_gap >= 0.9 * fortin_gap, (
        f"CovBat species gap {chen_gap:.3f} < 0.9 x Fortin gap {fortin_gap:.3f}"
    )


def test_covbat_changes_held_out_rows_only_along_the_retained_pcs():
    """On a held-out row, CovBat differs from Fortin only along the retained PCs.

    Fewer training rows than features, so held-out residuals extend outside the PC span.
    """
    X, batch, disc = simulate_species_covbat_data(n_per_batch=15, n_features=60)
    held_out = np.arange(len(X)) % 5 == 2
    assert (~held_out).sum() < X.shape[1]
    kwargs = {"batch": batch, "discrete_covariates": disc}
    chen = ComBat(method="chen", **kwargs).fit(X[~held_out])
    fortin = ComBat(method="fortin", **kwargs).fit(X[~held_out])

    change = chen.transform(X[held_out]).to_numpy() - fortin.transform(X[held_out]).to_numpy()
    model = chen._model
    retained = model._covbat_pca.components_[: model._covbat_n_pc]
    off_retained = change - (change @ retained.T) @ retained
    np.testing.assert_allclose(off_retained, 0.0, atol=1e-8)


@pytest.mark.parametrize("method", ["chen", "covbat_gam"])
def test_covbat_held_out_correction_does_not_depend_on_the_other_rows(method):
    """A held-out row gets the same correction alone, in a block, or with the training rows.

    Held-out rows exclude the age extremes, so their ages stay inside the spline bounds.
    """
    X, batch, disc = simulate_species_covbat_data()
    kwargs = {"batch": batch, "discrete_covariates": disc, "method": method}
    if method == "covbat_gam":
        kwargs["continuous_covariates"] = pd.DataFrame(
            {"age": np.linspace(20, 80, len(X))}, index=X.index
        )
        kwargs["spline_df"] = 5
    held_out = np.arange(len(X)) % 5 == 2
    X_train, X_test = X[~held_out], X[held_out]

    combat = ComBat(**kwargs).fit(X_train)
    block = combat.transform(X_test)
    row_by_row = pd.concat([combat.transform(X_test.iloc[[i]]) for i in range(len(X_test))])
    alongside_train = combat.transform(X).loc[X_test.index]

    np.testing.assert_allclose(row_by_row.values, block.values, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(alongside_train.values, block.values, rtol=1e-10, atol=1e-10)


def test_covbat_without_covariates_decomposes_the_fortin_output():
    """Without covariates the residual PCA is the PCA of the Fortin output."""
    X, batch, _ = simulate_species_covbat_data()
    fortin = ComBat(batch=batch, method="fortin").fit_transform(X)
    reference = PCA(svd_solver="full").fit(fortin.to_numpy())
    model = ComBat(batch=batch, method="chen", covbat_cov_thresh=0.9).fit(X)._model
    covbat_pca = model._covbat_pca

    np.testing.assert_allclose(
        covbat_pca.explained_variance_, reference.explained_variance_, rtol=1e-10
    )
    expected_n_pc = int(np.argmax(np.cumsum(reference.explained_variance_ratio_) >= 0.9)) + 1
    assert model._covbat_n_pc == expected_n_pc
    n_pc = model._covbat_n_pc
    alignment = np.abs(np.sum(covbat_pca.components_[:n_pc] * reference.components_[:n_pc], axis=1))
    np.testing.assert_allclose(alignment, 1.0, atol=1e-8)


def test_unknown_discrete_covariate_column_at_transform_raises():
    """A covariate column absent at fit has no recorded levels to encode against."""
    X, batch, sex = _covariate_cohort()
    fitted = sex.to_frame(name="sex")
    combat = ComBat(batch=batch, method="fortin", discrete_covariates=fitted).fit(X)
    combat.discrete_covariates = fitted.rename(columns={"sex": "gender"})
    with pytest.raises(ValueError, match="was not present at fit"):
        combat.transform(X)

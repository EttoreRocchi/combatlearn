import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils import simulate_longitudinal_data

from combatlearn import ComBat, LongitudinalComBat
from combatlearn._mixed import fit_random_intercept
from combatlearn.core import ComBatModel
from combatlearn.transductive import TransductiveComBat

pytestmark = pytest.mark.filterwarnings(
    "ignore:method='longitudinal' on the generic ComBat transformer:DeprecationWarning"
)


def _fit_kwargs():
    X, batch, subject, time = simulate_longitudinal_data()
    return X, {
        "batch": batch,
        "subject_id": subject,
        "time_covariate": time,
        "method": "longitudinal",
    }


def test_longitudinal_shape_index_columns_preserved():
    """Output keeps shape, index, and column names."""
    X, kwargs = _fit_kwargs()
    X.index = pd.Index([f"obs_{i}" for i in range(len(X))])
    kwargs["batch"].index = X.index
    kwargs["subject_id"].index = X.index
    kwargs["time_covariate"].index = X.index
    X.columns = [f"feat_{i}" for i in range(X.shape[1])]

    X_corr = ComBat(**kwargs).fit_transform(X)
    assert X_corr.shape == X.shape
    assert X_corr.index.equals(X.index)
    assert list(X_corr.columns) == list(X.columns)


def test_longitudinal_no_nan_or_inf():
    """Correction introduces no NaN/Inf."""
    X, kwargs = _fit_kwargs()
    X_corr = ComBat(**kwargs).fit_transform(X)
    assert not np.isnan(X_corr.values).any()
    assert not np.isinf(X_corr.values).any()


@pytest.mark.parametrize("mean_only", [True, False])
def test_longitudinal_mean_only(mean_only):
    """mean_only mode works."""
    X, kwargs = _fit_kwargs()
    X_corr = ComBat(mean_only=mean_only, **kwargs).fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


@pytest.mark.parametrize("parametric", [True, False])
def test_longitudinal_parametric_flag(parametric):
    """Both EB variants run."""
    X, kwargs = _fit_kwargs()
    X_corr = ComBat(parametric=parametric, **kwargs).fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_longitudinal_reduces_batch_variance():
    """Harmonization should reduce the fraction of variance explained by batch."""
    from combatlearn.inspection import batch_variance_explained

    X, kwargs = _fit_kwargs()
    cb = ComBat(**kwargs).fit(X)
    assert batch_variance_explained(cb, X) < cb._batch_var_before_


def test_longitudinal_requires_subject_id():
    """method='longitudinal' without subject_id raises."""
    X, batch, _subject, time = simulate_longitudinal_data()
    with pytest.raises(ValueError, match=r"requires `subject_id`"):
        ComBat(batch=batch, time_covariate=time, method="longitudinal").fit(X)


def test_subject_id_ignored_warning_for_other_methods():
    """Passing subject_id with a non-longitudinal method warns that it is ignored."""
    X, batch, subject, _time = simulate_longitudinal_data()
    with pytest.warns(UserWarning, match="only used when"):
        ComBat(batch=batch, subject_id=subject, method="johnson").fit(X)


def test_longitudinal_reference_batch_unchanged():
    """Reference-batch samples come out numerically identical."""
    X, batch, subject, time = simulate_longitudinal_data()
    ref = batch.iloc[0]
    cb = ComBat(
        batch=batch,
        subject_id=subject,
        time_covariate=time,
        method="longitudinal",
        reference_batch=ref,
    ).fit(X)
    X_corr = cb.transform(X)
    mask = (batch == ref).to_numpy()
    np.testing.assert_allclose(X_corr.values[mask], X.values[mask], rtol=0, atol=1e-8)


def test_longitudinal_clone_and_pipeline():
    """Wrapper clones and runs inside a Pipeline reproducibly."""
    X, kwargs = _fit_kwargs()
    pipe = Pipeline([("combat", ComBat(**kwargs)), ("scaler", StandardScaler())])
    X_corr = pipe.fit_transform(X)
    X_corr2 = clone(pipe).fit_transform(X)
    np.testing.assert_allclose(X_corr, X_corr2, rtol=1e-6, atol=1e-6)


def test_longitudinal_unseen_subject_falls_back_to_zero_re():
    """Transforming subjects unseen at fit (zero random intercept) must not crash."""
    X, batch, subject, time = simulate_longitudinal_data()
    model = ComBatModel(method="longitudinal").fit(
        X, batch=batch, subject_id=subject, time_covariate=time
    )
    subject_new = subject.map(lambda s: f"new_{s}")
    X_corr = model.transform(X, batch=batch, subject_id=subject_new, time_covariate=time)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_longitudinal_no_covariates():
    """Longitudinal with only a subject random intercept (no covariates) works."""
    X, batch, subject, _time = simulate_longitudinal_data()
    X_corr = ComBat(batch=batch, subject_id=subject, method="longitudinal").fit_transform(X)
    assert X_corr.shape == X.shape
    assert not np.isnan(X_corr.values).any()


def test_longitudinal_collinear_discrete_covariate_warns():
    """A discrete covariate collinear with batch should warn for the longitudinal method too."""
    X, batch, subject, _time = simulate_longitudinal_data()
    disc_collinear = batch.to_frame("cov")
    with pytest.warns(UserWarning, match="rank-deficient"):
        ComBat(
            batch=batch,
            subject_id=subject,
            discrete_covariates=disc_collinear,
            method="longitudinal",
        ).fit(X)


def test_longitudinal_collinear_time_covariate_warns():
    """A time covariate collinear with batch should also trigger the rank-deficiency warning."""
    X, batch, subject, _time = simulate_longitudinal_data()
    time_collinear = batch.map({"A": 0.0, "B": 1.0, "C": 2.0}).rename("visit")
    with pytest.warns(UserWarning, match="rank-deficient"):
        ComBat(
            batch=batch,
            subject_id=subject,
            time_covariate=time_collinear,
            method="longitudinal",
        ).fit(X)


def test_random_intercept_recovers_fixed_effects():
    """The REML fitter recovers known fixed effects on simulated data."""
    rng = np.random.default_rng(0)
    n_subjects, n_times = 60, 4
    n = n_subjects * n_times
    group_idx = np.repeat(np.arange(n_subjects), n_times).astype(np.intp)
    n_k = np.bincount(group_idx).astype(np.float64)

    x1 = rng.standard_normal(n)
    beta_true = np.array([2.0, 1.5])
    u = rng.normal(0.0, 1.0, n_subjects)
    eps = rng.normal(0.0, 1.0, n)
    y = beta_true[0] + beta_true[1] * x1 + u[group_idx] + eps

    X = np.column_stack([np.ones(n), x1])
    beta, _blup, sigma2, lam_opt = fit_random_intercept(X, y[:, None], group_idx, n_subjects, n_k)

    np.testing.assert_allclose(beta[:, 0], beta_true, rtol=0, atol=0.25)
    assert 0.5 < sigma2[0] < 2.0
    assert lam_opt[0] > 0.0


def test_random_intercept_matches_statsmodels():
    """Cross-check REML estimates against statsmodels.MixedLM."""
    sm = pytest.importorskip("statsmodels.api")

    rng = np.random.default_rng(1)
    n_subjects, n_times = 60, 5
    n = n_subjects * n_times
    group_idx = np.repeat(np.arange(n_subjects), n_times).astype(np.intp)
    n_k = np.bincount(group_idx).astype(np.float64)

    x1 = rng.standard_normal(n)
    u = rng.normal(0.0, 1.2, n_subjects)
    eps = rng.normal(0.0, 0.9, n)
    y = 1.0 + 0.8 * x1 + u[group_idx] + eps

    X = np.column_stack([np.ones(n), x1])
    beta, _blup, sigma2, lam_opt = fit_random_intercept(X, y[:, None], group_idx, n_subjects, n_k)
    tau2 = lam_opt[0] * sigma2[0]

    res = sm.MixedLM(endog=y, exog=X, groups=group_idx).fit(reml=True, method="lbfgs")
    sm_tau2 = float(np.asarray(res.cov_re)[0, 0])

    np.testing.assert_allclose(beta[:, 0], res.fe_params, rtol=0.02, atol=0.02)
    np.testing.assert_allclose(sigma2[0], res.scale, rtol=0.05)
    np.testing.assert_allclose(tau2, sm_tau2, rtol=0.12, atol=0.05)


def _long_kwargs():
    """``(X, kwargs)`` for the inductive ``LongitudinalComBat``."""
    X, batch, subject, time = simulate_longitudinal_data()
    return X, {"batch": batch, "subject_id": subject, "time_covariate": time}


def test_longitudinal_combat_is_exported():
    """LongitudinalComBat is importable from the package root."""
    import combatlearn

    assert combatlearn.LongitudinalComBat is LongitudinalComBat
    assert "LongitudinalComBat" in combatlearn.__all__


def test_longitudinal_combat_matches_transductive_whole_cohort():
    """Whole-cohort fit_transform is bit-identical to TransductiveComBat's."""
    X, kwargs = _long_kwargs()
    inductive = LongitudinalComBat(**kwargs).fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        transductive = TransductiveComBat(method="longitudinal", **kwargs).fit_transform(X)
    np.testing.assert_array_equal(inductive.values, transductive.values)


def test_longitudinal_combat_matches_deprecated_combat_path():
    """Whole-cohort fit_transform is bit-identical to ComBat(method='longitudinal')."""
    X, kwargs = _long_kwargs()
    inductive = LongitudinalComBat(**kwargs).fit_transform(X)
    deprecated = ComBat(method="longitudinal", **kwargs).fit_transform(X)
    np.testing.assert_array_equal(inductive.values, deprecated.values)


def test_longitudinal_combat_blup_reused_for_seen_zero_for_unseen():
    """Seen subjects reuse their frozen BLUP; unseen ones get exactly zero.

    With ``mean_only=False`` the seen/unseen gap is ``blup * (1 - 1/sqrt(delta_star))``.
    """
    X, batch, subject, time = simulate_longitudinal_data()
    est = LongitudinalComBat(batch=batch, subject_id=subject, time_covariate=time).fit(X)
    out_seen = est.transform(X)

    est.set_params(subject_id=subject.map("ghost_a_{}".format))
    out_unseen_a = est.transform(X)
    est.set_params(subject_id=subject.map("ghost_b_{}".format))
    out_unseen_b = est.transform(X)
    np.testing.assert_array_equal(out_unseen_a.values, out_unseen_b.values)

    model = est._model
    levels = list(model._re_subject_levels)
    blup_rows = model._re_blup[[levels.index(s) for s in subject]]
    assert np.abs(blup_rows).max() > 0

    batch_pos = {lvl: i for i, lvl in enumerate(model._batch_levels)}
    delta_rows = model._delta_star[[batch_pos[b] for b in batch]]
    expected = blup_rows * (1.0 - 1.0 / np.sqrt(delta_rows))
    np.testing.assert_allclose(
        out_seen.values - out_unseen_a.values, expected, rtol=1e-9, atol=1e-9
    )


def test_longitudinal_combat_holdout_subjects_are_transformable():
    """A held-out visit of a subject seen at fit transforms without refitting."""
    X, batch, subject, time = simulate_longitudinal_data()
    train = (time < 2.0).to_numpy()
    est = LongitudinalComBat(batch=batch, subject_id=subject, time_covariate=time).fit(X[train])
    out = est.transform(X[~train])
    assert out.shape == X[~train].shape
    assert np.isfinite(out.values).all()


def test_longitudinal_combat_on_unseen_subject_error_raises():
    """on_unseen_subject='error' raises and names the unseen subjects."""
    X, batch, subject, time = simulate_longitudinal_data()
    est = LongitudinalComBat(
        batch=batch, subject_id=subject, time_covariate=time, on_unseen_subject="error"
    ).fit(X)
    est.set_params(subject_id=subject.map("ghost_{}".format))
    with pytest.raises(ValueError, match=r"were not seen at fit"):
        est.transform(X)


def test_longitudinal_combat_on_unseen_subject_error_passes_when_all_seen():
    """on_unseen_subject='error' is a no-op when the precondition holds."""
    X, kwargs = _long_kwargs()
    est = LongitudinalComBat(on_unseen_subject="error", **kwargs).fit(X)
    population = LongitudinalComBat(**kwargs).fit(X)
    np.testing.assert_array_equal(est.transform(X).values, population.transform(X).values)


def test_longitudinal_combat_invalid_on_unseen_subject_raises():
    """An unrecognized policy is rejected at fit."""
    X, kwargs = _long_kwargs()
    with pytest.raises(ValueError, match="on_unseen_subject="):
        LongitudinalComBat(on_unseen_subject="ignore", **kwargs).fit(X)


def test_longitudinal_combat_requires_subject_id():
    """subject_id=None is rejected with a message pointing at Fortin."""
    X, batch, _subject, time = simulate_longitudinal_data()
    with pytest.raises(ValueError, match=r"requires `subject_id`"):
        LongitudinalComBat(batch=batch, subject_id=None, time_covariate=time).fit(X)


def test_longitudinal_combat_emits_no_deprecation_warning():
    """LongitudinalComBat emits no DeprecationWarning."""
    X, kwargs = _long_kwargs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        LongitudinalComBat(**kwargs).fit_transform(X)


def test_longitudinal_combat_clone_and_pipeline():
    """The estimator clones and runs inside a Pipeline reproducibly."""
    X, kwargs = _long_kwargs()
    pipe = Pipeline([("combat", LongitudinalComBat(**kwargs)), ("scaler", StandardScaler())])
    out = pipe.fit_transform(X)
    np.testing.assert_allclose(out, clone(pipe).fit_transform(X), rtol=1e-10, atol=1e-10)


def test_combat_longitudinal_deprecation_points_to_longitudinal_combat():
    """ComBat(method='longitudinal') warns toward LongitudinalComBat, not TransductiveComBat."""
    X, kwargs = _long_kwargs()
    with pytest.warns(DeprecationWarning, match="LongitudinalComBat") as record:
        ComBat(method="longitudinal", **kwargs).fit(X)
    assert not any("TransductiveComBat" in str(w.message) for w in record)


def test_transductive_longitudinal_deprecation_points_to_longitudinal_combat():
    """TransductiveComBat(method='longitudinal') deprecation names LongitudinalComBat."""
    X, kwargs = _long_kwargs()
    with pytest.warns(DeprecationWarning, match="LongitudinalComBat"):
        TransductiveComBat(method="longitudinal", **kwargs).fit(X)


def test_longcombat_alias_matches_longitudinal():
    """The 'longcombat' alias produces the same result as method='longitudinal'."""
    X, batch, subject, time = simulate_longitudinal_data()
    out_alias = ComBat(
        batch=batch, subject_id=subject, time_covariate=time, method="longcombat"
    ).fit_transform(X)
    out_canon = ComBat(
        batch=batch, subject_id=subject, time_covariate=time, method="longitudinal"
    ).fit_transform(X)
    np.testing.assert_allclose(out_alias.values, out_canon.values, rtol=1e-10, atol=1e-10)

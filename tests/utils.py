import numpy as np
import pandas as pd


def simulate_gam_data(
    n_samples=300,
    n_features=12,
    shift=3.0,
    signal=4.0,
    confound=True,
    nonlinear=True,
    random_state=0,
):
    """Data with an additive batch shift over a (non)linear age effect.

    Returns ``(X, batch, cont, oracle)``; ``oracle`` is the batch-free signal (effect + noise).
    ``confound=True`` assigns batches by age tertiles; ``nonlinear=True`` makes the age
    effect a full-period sinusoid, which ``fortin`` cannot separate from batch.
    """
    rng = np.random.default_rng(random_state)
    age = rng.uniform(20, 80, n_samples)
    if confound:
        q = np.quantile(age, [1 / 3, 2 / 3])
        batch_arr = np.where(age < q[0], "A", np.where(age < q[1], "B", "C"))
    else:
        batch_arr = rng.choice(["A", "B", "C"], n_samples)
    curve = np.sin((age - 20) / 60 * 2 * np.pi) if nonlinear else (age - 50) / 30
    loadings = rng.uniform(0.5, 1.5, n_features)
    effect = signal * np.outer(curve, loadings)
    noise = rng.standard_normal((n_samples, n_features))
    shifts = {"A": shift, "B": -shift, "C": 0.3 * shift}
    batch_shift = np.array([shifts[b] for b in batch_arr])[:, None]

    X = pd.DataFrame(effect + noise + batch_shift)
    idx = X.index
    batch = pd.Series(batch_arr, index=idx, name="batch")
    cont = pd.DataFrame({"age": age}, index=idx)
    oracle = pd.DataFrame(effect + noise, index=idx)
    return X, batch, cont, oracle


def simulate_covbat_data(
    n_samples=300,
    n_features=12,
    n_components=4,
    shift=3.0,
    signal=3.0,
    cov_strength=2.0,
    random_state=0,
):
    """GAM-style data with a batch effect in the covariance, for CovBat.

    Returns ``(X, batch, cont)``. On top of a nonlinear age effect, a rank-``n_components``
    block scaled by ``cov_strength``, independent of age and rescaled per batch, gives the
    batches different covariances, so CovBat retains several PCs.
    """
    rng = np.random.default_rng(random_state)
    age = rng.uniform(20, 80, n_samples)
    batch_arr = rng.choice(["A", "B", "C"], n_samples)

    curve = np.sin((age - 20) / 60 * 2 * np.pi)
    mean_signal = signal * np.outer(curve, rng.uniform(0.5, 1.5, n_features))

    loadings = rng.standard_normal((n_components, n_features))
    scores = rng.standard_normal((n_samples, n_components))
    batch_scale = {
        "A": np.linspace(2.0, 0.5, n_components),
        "B": np.linspace(0.5, 2.0, n_components),
        "C": np.ones(n_components),
    }
    scores = scores * np.array([batch_scale[b] for b in batch_arr])
    correlated = cov_strength * (scores @ loadings)

    noise = rng.standard_normal((n_samples, n_features))
    shifts = {"A": shift, "B": -shift, "C": 0.3 * shift}
    batch_shift = np.array([shifts[b] for b in batch_arr])[:, None]

    X = pd.DataFrame(mean_signal + correlated + noise + batch_shift)
    idx = X.index
    batch = pd.Series(batch_arr, index=idx, name="batch")
    cont = pd.DataFrame({"age": age}, index=idx)
    return X, batch, cont


def simulate_species_covbat_data(
    n_per_batch=80,
    n_features=40,
    n_species_features=10,
    species_effect=2.0,
    n_components=4,
    random_state=0,
):
    """CovBat data with a discrete covariate that is unbalanced across batches.

    Returns ``(X, batch, disc)``: three batches of ``n_per_batch`` samples, each with its
    own location shift and covariance. ``disc["species"]`` is 80%, 50% and 20% ``"S1"`` in
    batches A, B and C; ``"S2"`` adds ``species_effect`` to the first ``n_species_features``.
    """
    rng = np.random.default_rng(random_state)
    s1_share = {"A": 0.8, "B": 0.5, "C": 0.2}
    batch_arr = []
    species_arr = []
    for lvl, share in s1_share.items():
        n_s1 = round(share * n_per_batch)
        batch_arr += [lvl] * n_per_batch
        species_arr += ["S1"] * n_s1 + ["S2"] * (n_per_batch - n_s1)
    batch_np = np.asarray(batch_arr)
    species_np = np.asarray(species_arr)
    n_samples = len(batch_np)

    centers = {"A": 2.0, "B": -2.0, "C": 0.6}
    shifts = {lvl: rng.normal(centers[lvl], 0.5, n_features) for lvl in s1_share}
    location = np.array([shifts[b] for b in batch_np])

    loadings = rng.standard_normal((n_components, n_features))
    batch_scale = {
        "A": np.linspace(2.0, 0.5, n_components),
        "B": np.linspace(0.5, 2.0, n_components),
        "C": np.ones(n_components),
    }
    scores = rng.standard_normal((n_samples, n_components))
    correlated = (scores * np.array([batch_scale[b] for b in batch_np])) @ loadings

    noise = rng.standard_normal((n_samples, n_features))
    species = np.zeros((n_samples, n_features))
    species[species_np == "S2", :n_species_features] = species_effect

    X = pd.DataFrame(
        species + location + correlated + noise,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    idx = X.index
    batch = pd.Series(batch_np, index=idx, name="batch")
    disc = pd.DataFrame({"species": species_np}, index=idx)
    return X, batch, disc


def within_batch_species_gap(X, batch, disc, n_species_features=10):
    """Mean within-batch S2 - S1 gap on the species features (true value: ``species_effect``)."""
    values = np.asarray(X)[:, :n_species_features]
    b = np.asarray(batch)
    s = np.asarray(disc["species"])
    gaps = [
        (values[(b == lvl) & (s == "S2")].mean(0) - values[(b == lvl) & (s == "S1")].mean(0)).mean()
        for lvl in np.unique(b)
    ]
    return float(np.mean(gaps))


def residual_batch_gap(X, batch, disc):
    """Mean |batch mean| after centering each species on its pooled mean (~0 when harmonized)."""
    values = np.array(X, dtype=float)
    b = np.asarray(batch)
    s = np.asarray(disc["species"])
    for sp in np.unique(s):
        values[s == sp] -= values[s == sp].mean(0)
    batch_means = np.vstack([values[b == lvl].mean(0) for lvl in np.unique(b)])
    return float(np.abs(batch_means).mean())


def simulate_data(n_samples=150, n_features=25, shift=3.0, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    batches = np.repeat(
        ["A", "B", "C"],
        [n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)],
    )
    shifts = {"A": shift, "B": -shift, "C": 0.3 * shift}
    for i, b in enumerate(batches):
        X[i] += shifts[b]
    X = pd.DataFrame(X)
    batch_ser = pd.Series(batches, index=X.index, name="batch")
    return X, batch_ser


def simulate_covariate_data(random_state=0):
    X, batch = simulate_data()
    rng = np.random.default_rng(random_state)
    disc = pd.DataFrame({"diag": rng.choice(["diag_1", "diag_2"], size=len(X))}, index=X.index)
    age = rng.normal(50, 12, size=len(X))
    cont = pd.DataFrame({"age": age}, index=X.index)
    X.loc[disc["diag"] == "diag_1"] += 1.5
    X += (age[:, None] - age.mean()) * 0.04
    return X, batch, disc, cont


def simulate_nested_data(
    n_samples=240,
    n_features=15,
    n_vars=3,
    shift=2.5,
    signal=2.0,
    latent_split=False,
    random_state=0,
):
    """Multi-batch data with an additive shift per batch variable, for NestedComBat.

    Returns ``(X, batch, disc, cont)``; ``batch`` has one column per batch variable (the
    first ``n_vars`` of site, scanner, protocol) over a linear age effect.
    ``latent_split=True`` adds a hidden bimodal grouping to every feature, for GMM ComBat.
    """
    rng = np.random.default_rng(random_state)
    names = ["site", "scanner", "protocol"][:n_vars]
    levels = {"site": ["S1", "S2"], "scanner": ["GE", "SI", "PH"], "protocol": ["P1", "P2"]}

    total_shift = np.zeros((n_samples, n_features))
    batch_cols = {}
    for name in names:
        assign = rng.choice(levels[name], n_samples)
        batch_cols[name] = assign
        shift_map = {lvl: rng.uniform(-shift, shift) for lvl in levels[name]}
        per_sample = np.array([shift_map[a] for a in assign])[:, None]
        total_shift += per_sample * rng.uniform(0.5, 1.5, n_features)

    if latent_split:
        group = rng.integers(0, 2, n_samples)
        total_shift += (2.0 * group)[:, None] * rng.uniform(0.5, 1.5, n_features)

    age = rng.uniform(20, 80, n_samples)
    effect = signal * np.outer((age - 50) / 30, rng.uniform(0.5, 1.5, n_features))
    noise = rng.standard_normal((n_samples, n_features))

    X = pd.DataFrame(effect + noise + total_shift, columns=[f"feat_{i}" for i in range(n_features)])
    idx = X.index
    batch = pd.DataFrame({k: pd.Series(v, index=idx) for k, v in batch_cols.items()})
    disc = pd.DataFrame({"sex": rng.choice(["M", "F"], n_samples)}, index=idx)
    cont = pd.DataFrame({"age": age}, index=idx)
    return X, batch, disc, cont


def simulate_longitudinal_data(
    n_subjects=40, n_times=3, n_features=20, shift=3.0, re_sd=1.5, random_state=0
):
    """Repeated-measures data where each subject is observed across batches over time.

    Returns (X, batch, subject, time). Subjects carry a random intercept and span
    multiple batches across their time points, which identifies the mixed model.
    """
    rng = np.random.default_rng(random_state)
    n = n_subjects * n_times
    batch_levels = ["A", "B", "C"]
    shifts = {"A": shift, "B": -shift, "C": 0.3 * shift}

    subject_re = rng.normal(0.0, re_sd, size=(n_subjects, n_features))
    X = np.empty((n, n_features))
    subjects, times, batches = [], [], []
    row = 0
    for s in range(n_subjects):
        for t in range(n_times):
            b = batch_levels[(s + t) % len(batch_levels)]
            X[row] = rng.standard_normal(n_features) + subject_re[s] + shifts[b] + 0.5 * t
            subjects.append(f"subj_{s}")
            times.append(float(t))
            batches.append(b)
            row += 1

    X = pd.DataFrame(X)
    idx = X.index
    batch = pd.Series(batches, index=idx, name="batch")
    subject = pd.Series(subjects, index=idx, name="subject")
    time = pd.Series(times, index=idx, name="time")
    return X, batch, subject, time

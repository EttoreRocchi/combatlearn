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
    """Data with an additive batch shift over a (non)linear continuous-covariate effect.

    Returns ``(X, batch, cont, oracle)`` where ``oracle`` is the batch-free signal
    (true covariate effect + noise) that a perfect harmonizer should recover.

    When ``confound=True`` the batch label is assigned from age tertiles, so the
    age-driven signal is collinear with batch. With ``nonlinear=True`` (a full-period
    sinusoid) a linear covariate model (``fortin``) cannot separate the nonlinear age
    effect from the batch effect, while a spline model (``gam``) can.
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


def simulate_data(n_samples=150, n_features=25, shift=3.0, random_state=0):
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    batches = np.repeat(
        ["A", "B", "C"],
        [n_samples // 3, n_samples // 3, n_samples - 2 * (n_samples // 3)],
    )
    shifts = {"A": shift, "B": -shift, "C": 0.3 * shift}  # to shift the means
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
    X.loc[disc["diag"] == "diag_1"] += 1.5  # to add effect in diag
    X += (age[:, None] - age.mean()) * 0.04  # to add effect in age
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

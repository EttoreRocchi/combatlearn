import numpy as np
import pandas as pd


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

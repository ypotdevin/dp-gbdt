import joblib
import numpy as np
import pandas as pd

import dpgbdt
from typing import ItemsView


def abalone_scaling_factors(items: ItemsView) -> pd.DataFrame:
    scaling_factorss = joblib.Parallel(n_jobs=16)(
        joblib.delayed(_worker)(error_vectors, 20.0) for (_, error_vectors) in items
    )
    scaling_factors = np.array(scaling_factorss).max(axis=0)

    pbs = np.linspace(1e-3, 1.0, 100)
    rels = np.array([1e-7, 1e-6, 1e-5, 1e-4])
    bs = np.logspace(-6, -1, 600)
    pbs, rels, bs = np.meshgrid(pbs, rels, bs, indexing="ij")
    entries = dict(
        privacy_budget=pbs.reshape(-1),
        relaxation=rels.reshape(-1),
        beta=bs.reshape(-1),
        scaling_factor=scaling_factors.reshape(-1),
    )
    df = pd.DataFrame(entries)
    return df


def _worker(error_vectors: np.ndarray, upper_bound: float) -> np.ndarray:
    pbs = np.linspace(1e-3, 1.0, 100)
    rels = np.array([1e-7, 1e-6, 1e-5, 1e-4])
    bs = np.logspace(-6, -1, 600)
    pbs, rels, bs = np.meshgrid(pbs, rels, bs, indexing="ij")
    scaling_factorss = np.array(
        [
            _scaling_factors(error_vector, pbs, rels, bs, upper_bound)
            for error_vector in error_vectors
        ]
    )
    scaling_factors = scaling_factorss.max(axis=0)
    return scaling_factors


def _scaling_factors(
    errors: np.ndarray,
    privacy_budgets: np.ndarray,
    relaxations: np.ndarray,
    betas: np.ndarray,
    upper_bound: float,
) -> np.ndarray:
    beta_smooth_sensitivities = np.array(
        [
            dpgbdt.py_beta_smooth_sensitivity(errors, beta, upper_bound)
            for beta in betas[0, 0, :]
        ]
    )

    alphas = privacy_budgets + betas - (np.exp(betas) - 1.0) * np.log(1.0 / relaxations)
    alphas = np.where(alphas <= 0.0, np.nan, alphas)
    scaling_factors = beta_smooth_sensitivities[np.newaxis, np.newaxis, :] / alphas
    return scaling_factors.reshape(-1)


if __name__ == "__main__":
    with np.load("abalone_bun_steinke_20221107_error_vectors.npz") as arrays:
        # arr = arrays["abalone_bun_steinke_20221107_feature-grid.17610799.log"]
        # d = {"abalone_bun_steinke_20221107_feature-grid.17610799.log": arr}
        df = abalone_scaling_factors(arrays.items())
        df.to_csv("scaling_factors.csv")

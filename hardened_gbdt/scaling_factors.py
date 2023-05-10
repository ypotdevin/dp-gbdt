"""
The Bun & Steinke tree scoring mechanism requires to set a β parameter
a priori. This module helps estimating a good β, by generating tuples of
(β, ε, δ, scaling_factor), i.e. the smoothness parameter β, the privacy
budget ε, the relaxation parameter δ and the resulting noise scaling
factor according to Bun & Steinke's derivations.
"""
from typing import ItemsView

import joblib
import numpy as np
import pandas as pd

import dpgbdt


def scaling_factors(
    errors: np.ndarray,
    upper_bound: float,
    privacy_budgets: np.ndarray,
    relaxations: np.ndarray,
    betas: np.ndarray,
) -> pd.DataFrame:
    """Given an array of errors, an upper bound on the errors, privacy
    budgets, relaxation parameters and smoothness parameters, return for
    each combination of privacy budget, relaxation parameter and
    smoothness parameter a scaling factor.

    Args:
        errors (np.ndarray): error vector, should be non-negative and
        ordered ascendingly.
        upper_bound (float): upper bound on errors (ideally, this is an
          a priori value instead of just errors.max())
        privacy_budgets (np.ndarray): array of shape (i,)
        relaxations (np.ndarray): array of shape (j,)
        betas (np.ndarray): array of shape (k,)

    Returns:
        pd.DataFrame: a DataFrame of length i * j * k
    """
    beta_smooth_sensitivities = np.array(
        [dpgbdt.py_beta_smooth_sensitivity(errors, beta, upper_bound) for beta in betas]
    )
    pbs, rels, bs = np.meshgrid(privacy_budgets, relaxations, betas, indexing="ij")
    alphas = pbs + bs - (np.exp(bs) - 1.0) * np.log(1.0 / rels)
    alphas = np.where(alphas <= 0.0, np.nan, alphas)
    scaling_factors = beta_smooth_sensitivities[np.newaxis, np.newaxis, :] / alphas
    return pd.DataFrame(
        dict(
            privacy_budget=pbs.reshape(-1),
            relaxation=rels.reshape(-1),
            beta=bs.reshape(-1),
            scaling_factor=scaling_factors.reshape(-1),
        )
    )


def scaling_factors_from_dataset(
    y: np.ndarray,
    privacy_budgets: np.ndarray,
    relaxations: np.ndarray,
    betas: np.ndarray,
) -> pd.DataFrame:
    """Like `scaling_factors`, but determine the error vector and upper
    bound from the target vector (y) of a training dataset.
    """
    errors = np.sort(np.abs(y - y.mean()))
    return scaling_factors(
        errors=errors,
        upper_bound=errors.max(),
        privacy_budgets=privacy_budgets,
        relaxations=relaxations,
        betas=betas,
    )


def abalone_scaling_factors(items: ItemsView, n_jobs: int = 32) -> pd.DataFrame:
    scaling_factorss = joblib.Parallel(n_jobs=n_jobs)(
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
            _scaling_factors(error_vector, upper_bound, pbs, rels, bs)
            for error_vector in error_vectors
        ]
    )
    scaling_factors = scaling_factorss.max(axis=0)
    return scaling_factors


def _scaling_factors(
    errors: np.ndarray,
    upper_bound: float,
    privacy_budgets: np.ndarray,
    relaxations: np.ndarray,
    betas: np.ndarray,
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
    from datasets.real import data_reader

    m = data_reader.metro_fit_arguments()
    pbs = np.linspace(1e-3, 1.0, 100)
    rels = np.array([1e-7, 1e-6, 1e-5, 1e-4])
    bs = np.logspace(-6, -1, 600)
    df = scaling_factors_from_dataset(
        y=m["y"],
        privacy_budgets=pbs,
        relaxations=rels,
        betas=bs,
    )
    df.to_csv("scaling_factors_metro.csv.zip")

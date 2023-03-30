import argparse
import logging
from itertools import chain
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn import model_selection

from probabilitybuckets_light import ProbabilityBuckets


def privacy_bucket_score_budget(
    scaling_factor_component: float,
    beta: float,
    relaxations: Iterable[float],
    n_trees_to_accept: int,
    log_level: int = logging.WARNING,
) -> np.ndarray:
    """Given alpha, beta, the number of trees to accept and an array of
    relaxations, return an array of privacy budgets, estimated via
    privacy buckets.

    Args:
        scaling_factor_component (float): …
        beta (float): a beta value. By beta we mean the beta from
        (alpha, beta)-admissible distributions.
        relaxations (Iterable[float]): the relaxation parameter (delta)
        from approximate DP. Provide multiple relaxations at once to
        save computation time in contrast to calling this function once
        for each single relaxation value.
        n_trees_to_accept (int): the number of trees to accept.
        log_level (int, optional): Defaults to logging.WARNING.

    Returns:
        np.ndarray: an array of epsilons corresponding to the relaxations.
        That means for a given `alpha`, `beta` `n_trees_to_accept` and
        `relaxations[i]`, `epsilons[i]` is the privacy budget which will
        be spent in total, when scoring and accepting `n_trees_to_accept`
        trees (via the `score_tree(...)` method).
    """
    loose_gaussian_compositional_bound = 5.3
    sigma = (
        loose_gaussian_compositional_bound
        * float(np.sqrt(n_trees_to_accept))
        / scaling_factor_component
    )
    standard_gaussian_tail_cufoff = 40.0
    tail_cutoff = standard_gaussian_tail_cufoff * sigma
    epsilon_arrays = []
    for factor in [1 + 5e-5, 1 + 5e-5, 1 + 5e-6, 1 + 5e-7]:
        try:
            epsilon_arrays.append(
                _privacy_bucket_score_budget(
                    scaling_factor_component,
                    beta,
                    relaxations,
                    n_trees_to_accept,
                    log_level,
                    sigma=sigma,
                    tail_cutoff=tail_cutoff,
                    factor=factor,
                    number_of_buckets=100_000,
                    sampling_rate=5_000_000,
                )
            )
        except ValueError:
            pass
    if epsilon_arrays:
        return np.nanmax(epsilon_arrays, axis=0)
    else:
        return np.array([np.nan for _ in relaxations])


def _privacy_bucket_score_budget(
    scaling_factor_component: float,
    beta: float,
    relaxations: Iterable[float],
    n_trees_to_accept: int,
    log_level: int,
    sigma: float,
    tail_cutoff: float,
    factor: float,
    number_of_buckets: int,
    sampling_rate: int,
) -> np.ndarray:
    support = np.linspace(
        -tail_cutoff / scaling_factor_component,
        tail_cutoff / scaling_factor_component,
        num=sampling_rate,
        endpoint=True,
    )
    dilated_sigma = sigma * np.exp(beta)

    # Important: the individual both distributions need to sum up to 1 exactly, respectively!
    reference_distribution = norm.pdf(support, loc=0, scale=sigma)
    reference_distribution = reference_distribution / np.sum(reference_distribution)
    dilated_distribution = norm.pdf(support, loc=1, scale=dilated_sigma)
    dilated_distribution = dilated_distribution / np.sum(dilated_distribution)

    privacybuckets = ProbabilityBuckets(
        number_of_buckets=number_of_buckets,
        factor=factor,
        dist1_array=dilated_distribution,
        dist2_array=reference_distribution,
        # caching makes re-evaluations faster. Can be turned off for some cases.
        caching_directory="./pb-cache",
        # how much we can put in the infty bucket before first squaring
        free_infty_budget=10 ** (-20),
        error_correction=True,
        logging_level=log_level,
    )
    privacybuckets_composed = privacybuckets.compose(n_trees_to_accept)

    # Print status summary
    privacybuckets_composed.print_state()

    epsilons = np.array(
        [
            privacybuckets_composed.eps_ADP_upper_bound(relaxation)
            for relaxation in relaxations
        ]
    )
    return epsilons


def _to_lines(
    scaling_factor_component: float,
    beta: float,
    relaxations: Iterable[float],
    n_trees_to_accept: int,
    epsilons: Iterable[float],
) -> list[dict[str, Any]]:
    return [
        dict(
            scaling_factor_component=scaling_factor_component,
            beta=beta,
            n_trees_to_accept=n_trees_to_accept,
            relaxation=relaxation,
            epsilon=epsilon,
        )
        for (relaxation, epsilon) in zip(relaxations, epsilons)
    ]


def _worker(beta, relaxations, n_trees_to_accept, scaling_factor_component):
    return _to_lines(
        scaling_factor_component,
        beta,
        relaxations,
        n_trees_to_accept,
        privacy_bucket_score_budget(
            scaling_factor_component, beta, relaxations, n_trees_to_accept
        ),
    )


def evaluate_search_space(
    search_space: model_selection.ParameterGrid, n_jobs: int
) -> pd.DataFrame:
    lines_list = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(**kwargs) for kwargs in search_space
    )
    df = pd.DataFrame(chain(*lines_list))  # type: ignore
    return df


def search_space_20230329() -> model_selection.ParameterGrid:
    parameter_grid = model_selection.ParameterGrid(
        dict(
            beta=np.linspace(start=1e-4, stop=1e-3, num=10),
            relaxations=[[1e-4, 1e-5, 1e-6, 1e-7]],
            n_trees_to_accept=[1, 2, 3, 5, 8, 10, 20, 50],
            scaling_factor_component=np.logspace(
                start=0, stop=2, num=(2 - 0) * 10, base=10.0
            ),
        )
    )
    return parameter_grid


def search_space_20230330() -> model_selection.ParameterGrid:
    parameter_grid = model_selection.ParameterGrid(
        dict(
            beta=np.linspace(start=1e-4, stop=1e-3, num=20),
            relaxations=[[1e-4, 1e-5, 1e-6, 1e-7]],
            n_trees_to_accept=[1, 2, 3, 5, 8, 10, 20, 50],
            scaling_factor_component=np.linspace(start=0, stop=10, num=100),
        )
    )
    return parameter_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform grid search over the parameter space."
    )
    parser.add_argument(
        "search_space",
        type=str,
        help="which search space to evaluate",
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default=None,
        help="path to output .csv file",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=-1,
        help="Number of CPU cores to use for hyperparameter search. Default: all cores.",
    )
    args = parser.parse_args()

    dispatcher = dict(
        search_space_20230329=search_space_20230329(),
    )
    df = evaluate_search_space(dispatcher[args.search_space], args.num_cores)
    df.to_csv(args.csv_filename)

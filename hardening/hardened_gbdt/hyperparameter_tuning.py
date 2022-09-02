import argparse
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from ray.tune.sklearn import TuneSearchCV

import dpgbdt


def tune(
    regressor,
    data_provider: Callable[[], Tuple[np.ndarray, np.ndarray, list[int], list[int]]],
    parameter_grid: dict[str, Any],
    label: str,
    n_trials: int,
    local_dir: str,
    n_jobs: Optional[int] = None,
    time_budget_s: Optional[int] = None,
    search_optimization="hyperopt",
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    X_train, y_train, cat_idx, num_idx = data_provider()
    tune_search = TuneSearchCV(
        regressor,
        parameter_grid,
        name=label,
        search_optimization=search_optimization,
        n_trials=n_trials,
        local_dir=local_dir,
        n_jobs=n_jobs,
        time_budget_s=time_budget_s,
        scoring="neg_root_mean_squared_error",
    )
    tune_search.fit(X_train, y_train, cat_idx=cat_idx, num_idx=num_idx)
    df = pd.DataFrame(tune_search.cv_results_)
    return (df, tune_search.best_params_)


def get_abalone() -> Tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Parse the abalone dataset.
    Returns:
      Any: X, y, cat_idx, num_idx
    """
    data = pd.read_csv(
        "./datasets/real/abalone.data",
        names=[
            "sex",
            "length",
            "diameter",
            "height",
            "whole weight",
            "shucked weight",
            "viscera weight",
            "shell weight",
            "rings",
        ],
    )
    data["sex"], _ = pd.factorize(data["sex"])
    y = data.rings.values.astype(float)
    del data["rings"]
    X = data.values.astype(float)
    cat_idx = [0]  # Sex
    num_idx = list(range(1, X.shape[1]))  # Other attributes
    return X, y, cat_idx, num_idx


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform Bayesian optimization over the hyperparameter space."
    )
    parser.add_argument(
        "label", type=str, help="The label of the experiment.",
    )
    parser.add_argument(
        "csvfilename",
        type=str,
        help="Name of the .csv file containing configurations and their performance.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="The number of configurations to test at most (if the time budget allows).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="./ray_results",
        help="In which folder to save intermediate results, checkpoints and logs.",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for hyperparameter search. Default: all cores.",
    )
    parser.add_argument(
        "--time-budget-s",
        type=int,
        default=3600 * 24,  # 24 hours
        help="How much time (in seconds) to spend (roughly) at most.",
    )
    args = parser.parse_args()
    return args


def abalone_parameter_grid():
    parameter_grid = dict(
        learning_rate=(0.0, 10.0),
        nb_trees=(1, 50),
        max_depth=(1, 10),
        l2_threshold=(0.01, 10.0),
        l2_lambda=(0.1, 10.0),
    )
    return parameter_grid


def baseline(args) -> pd.DataFrame:
    dfs = []
    for ensemble_budget in [0.1, 0.5, 1.0]:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [ensemble_budget]
        parameter_grid["tree_rejector"] = [
            dpgbdt.make_tree_rejector("constant", decision=False)
        ]
        df, _ = tune(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            label=args.label,
            n_trials=args.n_trials,
            local_dir=args.local_dir,
            n_jobs=args.num_cores,
            time_budget_s=args.time_budget_s,
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    return df


def baseline(args) -> pd.DataFrame:
    dfs = []
    for ensemble_budget in [0.1, 0.5, 1.0]:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [ensemble_budget]
        parameter_grid["tree_rejector"] = [
            dpgbdt.make_tree_rejector("constant", decision=False)
        ]
        df, _ = tune(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            label=args.label,
            n_trials=args.n_trials,
            local_dir=args.local_dir,
            n_jobs=args.num_cores,
            time_budget_s=args.time_budget_s,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def quantile_lin_comb(args) -> pd.DataFrame:
    dfs = []
    for ensemble_budget in [0.1, 0.5, 1.0]:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [ensemble_budget]
        tree_rejector = dpgbdt.make_tree_rejector(
            "quantile_linear_combination",
            qs=[0.50, 0.85, 0.95],
            coefficients=[0.23837927, 0.29496094, 0.16520499],
        )
        parameter_grid["tree_rejector"] = [tree_rejector]
        df, _ = tune(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            label=args.label,
            n_trials=args.n_trials,
            local_dir=args.local_dir,
            n_jobs=args.num_cores,
            time_budget_s=args.time_budget_s // len(ensemble_budget),
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def dp_rmse(args) -> pd.DataFrame:
    dfs = []
    for ensemble_budget in [0.1, 0.5, 1.0]:
        for rejection_budget in [0.01, 0.05, 0.1]:
            parameter_grid = abalone_parameter_grid()
            parameter_grid["privacy_budget"] = [ensemble_budget]
            tr_rng = dpgbdt.make_rng()
            rejector_params = dict(
                rejection_budget=rejection_budget,
                gamma=2.0,
                rng=tr_rng,
            )
            tree_rejectors = [
                dpgbdt.make_tree_rejector("dp_rmse", **{"U": U, **rejector_params})
                for U in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
            ]
            parameter_grid["tree_rejector"] = tree_rejectors
            df, _ = tune(
                dpgbdt.DPGBDTRegressor(),
                get_abalone,
                parameter_grid,
                label=args.label,
                n_trials=args.n_trials,
                local_dir=args.local_dir,
                n_jobs=args.num_cores,
                time_budget_s=args.time_budget_s,
            )
            dfs.append(df)
    df = pd.concat(dfs)
    return df

if __name__ == "__main__":
    args = parse_args()
    df = quantile_lin_comb(args)
    df.to_csv(args.csvfilename)

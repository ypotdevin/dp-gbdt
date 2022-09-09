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
        "experiment",
        type=str,
        help="Which experiment/benchmark to execute.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label of the experiment. Should be usable as a filename."
        " Defaults to <experiment>.",
    )
    parser.add_argument(
        "--csvfilename",
        type=str,
        default=None,
        help="Name of the .csv file containing configurations and their performance."
        " Defaults to <local_dir>/<label>.csv",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1000,
        help="The number of configurations to test at most per call to `TuneSearchCV.tune`"
        " (if the time budget allows).",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="In which folder to save intermediate results, checkpoints and logs."
        " Defaults to ~/share/dp-gbdt-evaluation/<label>",
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
        help="How much time (in seconds) to spend (roughly) at most in total."
        " Use `None` to deactivate limit.",
    )
    args = parser.parse_args()
    return args


def abalone_parameter_grid():
    parameter_grid = dict(
        learning_rate=(0.0, 10.0),
        n_trials=(1, 50),
        # Hopefully the hyperparameter optimizer "learns" to keep this
        # lower than `n_trials`
        n_trees_to_accept=(1, 50),
        max_depth=(1, 10),
        l2_threshold=(0.01, 10.0),
        l2_lambda=(0.1, 10.0),
    )
    return parameter_grid


def baseline(args) -> pd.DataFrame:
    dfs = []
    ensemble_budgets = [0.1, 0.5, 1.0, 2.0]
    for ensemble_budget in ensemble_budgets:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [ensemble_budget]
        parameter_grid["ensemble_rejector_budget_split"] = [1.0]
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
            time_budget_s=args.time_budget_s // len(ensemble_budgets),
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def quantile_lin_comb(args) -> pd.DataFrame:
    dfs = []
    ensemble_budgets = [0.1, 0.5, 1.0]
    for ensemble_budget in ensemble_budgets:
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
            time_budget_s=args.time_budget_s // len(ensemble_budgets),
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def dp_rmse(args) -> pd.DataFrame:
    dfs = []
    ensemble_budgets = [0.1, 0.5, 1.0, 2.0]
    for ensemble_budget in ensemble_budgets:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [ensemble_budget]
        parameter_grid["ensemble_rejector_budget_split"] = (1e-2, 1 - 1e-2)
        parameter_grid["tree_rejector"] = ["dp_rmse"]
        parameter_grid["tr_U"] = (0.1, 100.0)
        parameter_grid["tr_gamma"] = (1 + 1e-2, 10.0)

        df, _ = tune(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            label=args.label,
            n_trials=args.n_trials,
            local_dir=args.local_dir,
            n_jobs=args.num_cores,
            time_budget_s=args.time_budget_s // len(ensemble_budgets),
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def select_experiment(which: str) -> Callable[..., pd.DataFrame]:
    return dict(
        baseline=baseline,
        quantile_lin_comb=quantile_lin_comb,
        dp_rmse=dp_rmse,
    )[which]

if __name__ == "__main__":
    args = parse_args()
    if args.label is None:
        args.label = args.experiment
    if args.local_dir is None:
        args.local_dir = f"~/share/dp-gbdt-evaluation/"
    if args.csvfilename is None:
        args.csvfilename = f"{args.local_dir}/{args.label}.csv"
    experiment = select_experiment(args.experiment)
    df = experiment(args)
    df.to_csv(args.csvfilename)

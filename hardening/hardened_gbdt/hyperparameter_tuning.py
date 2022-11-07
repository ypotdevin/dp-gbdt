import argparse
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from ray.tune.sklearn import TuneSearchCV, TuneGridSearchCV
from sklearn.model_selection import GridSearchCV, RepeatedKFold

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
    cv=None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=2)
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
        cv=cv,
    )
    tune_search.fit(X_train, y_train, cat_idx=cat_idx, num_idx=num_idx)
    df = pd.DataFrame(tune_search.cv_results_)
    return df


def tune_grid(
    regressor,
    data_provider: Callable[[], Tuple[np.ndarray, np.ndarray, list[int], list[int]]],
    parameter_grid: dict[str, Any],
    label: str,
    local_dir: str,
    n_jobs: Optional[int] = None,
    time_budget_s: Optional[int] = None,
    cv=None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=2)
    X_train, y_train, cat_idx, num_idx = data_provider()
    tune_search = TuneGridSearchCV(
        regressor,
        parameter_grid,
        name=label,
        local_dir=local_dir,
        n_jobs=n_jobs,
        time_budget_s=time_budget_s,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    tune_search.fit(X_train, y_train, cat_idx=cat_idx, num_idx=num_idx)
    df = pd.DataFrame(tune_search.cv_results_)
    return df


def sklearn_grid(
    regressor,
    data_provider: Callable[[], Tuple[np.ndarray, np.ndarray, list[int], list[int]]],
    parameter_grid: dict[str, Any],
    n_jobs: Optional[int] = None,
    cv=None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=2)
    X_train, y_train, cat_idx, num_idx = data_provider()
    tune_search = GridSearchCV(
        regressor,
        parameter_grid,
        n_jobs=n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    tune_search.fit(X_train, y_train, cat_idx=cat_idx, num_idx=num_idx)
    df = pd.DataFrame(tune_search.cv_results_)
    return df


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
        "--budget",
        type=float,
        default=[],
        dest="privacy_budgets",
        action="append",
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
        default=-1,
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


def abalone_parameter_distribution():
    parameter_distr = dict(
        learning_rate=(0.0, 10.0),
        n_trees_to_accept=(1, 50),
        max_depth=(1, 10),
        l2_threshold=(0.01, 100.0),
        l2_lambda=(0.01, 100.0),
    )
    return parameter_distr


def abalone_parameter_grid():
    parameter_grid = dict(
        learning_rate=[0.1],
        max_depth=[1, 6],
        # 20.0 is the max. difference between any target value and the
        # average target value
        l2_threshold=np.linspace(0.5, 20.0, 10),
        l2_lambda=[0.1],
        n_trees_to_accept=[1, 2, 3, 5, 8, 10, 20, 50],
    )
    return parameter_grid


def abalone_parameter_grid_20221107():
    parameter_grid = dict(
        learning_rate=[0.1],
        max_depth=[1, 6],
        # 20.0 is the max. difference between any target value and the
        # average target value
        l2_threshold=np.linspace(0.5, 20.0, 10),
        l2_lambda=[0.1],
        n_trees_to_accept=[1, 2, 3, 5, 8, 10, 20, 50],
    )
    return parameter_grid


def abalone_parameter_dense_grid():
    parameter_grid = dict(
        learning_rate=[0.01, 0.1, 1.0],
        max_depth=[1, 2, 3, 4, 5, 6],
        # 20.0 is the max. difference between any target value and the
        # average target value
        l2_threshold=np.linspace(0.5, 20.0, 20),
        l2_lambda=np.linspace(0.1, 1.0, 10),
        n_trees_to_accept=[2, 3, 5, 8],
    )
    return parameter_grid


def baseline_template(args, grid: dict[str, Any]) -> pd.DataFrame:
    dfs = []
    total_budgets = args.privacy_budgets
    for ensemble_budget in total_budgets:
        parameter_grid = grid
        parameter_grid["training_variant"] = ["vanilla"]
        parameter_grid["privacy_budget"] = [ensemble_budget]
        parameter_grid["ensemble_rejector_budget_split"] = [1.0]
        parameter_grid["tree_rejector"] = [
            dpgbdt.make_tree_rejector("constant", decision=False)
        ]
        df = sklearn_grid(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            n_jobs=args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def baseline_grid(args) -> pd.DataFrame:
    return baseline_template(args, abalone_parameter_grid())


def baseline_grid_20221107(args) -> pd.DataFrame:
    return baseline_template(args, abalone_parameter_grid_20221107())


def baseline_dense_grid(args) -> pd.DataFrame:
    return baseline_template(args, abalone_parameter_dense_grid())


def dp_rmse_ts_template(args, grid: dict[str, Any]) -> pd.DataFrame:
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["tree_scorer"] = ["dp_rmse"]

        df = sklearn_grid(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            n_jobs=args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def dp_rmse_ts_grid(args) -> pd.DataFrame:
    grid = abalone_parameter_grid()
    grid["ensemble_rejector_budget_split"] = [0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_gamma"] = [2]
    return dp_rmse_ts_template(args, grid)


def dp_rmse_ts_grid_20221107(args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221107()
    grid["ensemble_rejector_budget_split"] = [0.2, 0.4, 0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.0001, 0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.01, 0.1, 0.2, 0.4]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_gamma"] = [2]
    return dp_rmse_ts_template(args, grid)


def dp_quantile_ts_grid(args) -> pd.DataFrame:
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["ensemble_rejector_budget_split"] = [0.6, 0.75, 0.9]
        parameter_grid["tree_scorer"] = ["dp_quantile"]
        parameter_grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
        parameter_grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
        parameter_grid["ts_shift"] = [0.0]
        parameter_grid["ts_scale"] = [0.79]
        parameter_grid["ts_upper_bound"] = parameter_grid["l2_threshold"]

        df = sklearn_grid(
            dpgbdt.DPGBDTRegressor(ts_qs=[0.5, 0.90, 0.95]),
            get_abalone,
            parameter_grid,
            n_jobs=args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def select_experiment(which: str) -> Callable[..., pd.DataFrame]:
    return dict(
        baseline_grid=baseline_grid,
        baseline_grid_20221107=baseline_grid_20221107,
        baseline_dense_grid=baseline_dense_grid,
        dp_rmse_ts_grid=dp_rmse_ts_grid,
        dp_rmse_ts_grid_20221107=dp_rmse_ts_grid_20221107,
        dp_quantile_ts_grid=dp_quantile_ts_grid,
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

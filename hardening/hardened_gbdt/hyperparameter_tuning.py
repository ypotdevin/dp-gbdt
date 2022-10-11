import argparse
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from ray.tune.sklearn import TuneSearchCV, TuneGridSearchCV
from sklearn.model_selection import RepeatedKFold

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
    return (df, tune_search.best_params_)


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
        "experiment", type=str, help="Which experiment/benchmark to execute.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=[0.1, 0.5, 1.0, 2.0],
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
        learning_rate=np.logspace(-6, 1, 8),
        max_depth=[1, 3, 6],
        # 20.0 is the max. difference between any target value and the
        # average target value
        l2_threshold=np.linspace(0.5, 20.0, 10),
        l2_lambda=[0.1, 1],
        n_trees_to_accept=[2, 3, 5, 8],
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


def dp_rmse_ts(args) -> pd.DataFrame:
    dfs = []
    ensemble_budgets = [0.1, 0.5, 1.0, 2.0]
    for ensemble_budget in ensemble_budgets:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [ensemble_budget]
        parameter_grid["ensemble_rejector_budget_split"] = (1e-2, 1 - 1e-2)
        parameter_grid["tree_scorer"] = ["dp_rmse"]
        parameter_grid["dp_argmax_privacy_budget"] = (1e-2, 1 - 1e-2)
        parameter_grid["dp_argmax_stopping_prob"] = (1e-2, 1 - 1e-2)
        parameter_grid["ts_upper_bound"] = (0.1, 100.0)
        parameter_grid["ts_gamma"] = (1e-2, 1 - 1e-2)

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


def dp_rmse_ts_grid(args) -> pd.DataFrame:
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["ensemble_rejector_budget_split"] = [0.6, 0.7, 0.8, 0.9]
        parameter_grid["tree_scorer"] = ["dp_rmse"]
        parameter_grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
        parameter_grid["dp_argmax_stopping_prob"] = [0.01, 0.1, 0.2, 0.5]
        parameter_grid["ts_upper_bound"] = parameter_grid["l2_threshold"]
        parameter_grid["ts_gamma"] = [2, 5]

        df, _ = tune_grid(
            dpgbdt.DPGBDTRegressor(),
            get_abalone,
            parameter_grid,
            label=args.label,
            local_dir=args.local_dir,
            n_jobs=args.num_cores,
            time_budget_s=args.time_budget_s // len(total_budgets),
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def dp_quantile_ts(args) -> pd.DataFrame:
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = abalone_parameter_grid()
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["ensemble_rejector_budget_split"] = (1e-2, 1 - 1e-2)
        parameter_grid["tree_scorer"] = ["dp_quantile"]
        parameter_grid["dp_argmax_privacy_budget"] = (1e-2, 1 - 1e-2)
        parameter_grid["dp_argmax_stopping_prob"] = (1e-2, 1 - 1e-2)
        parameter_grid["ts_shift"] = [0.0]
        parameter_grid["ts_scale"] = [0.79]
        parameter_grid["ts_upper_bound"] = (0.1, 100.0)

        df, _ = tune(
            dpgbdt.DPGBDTRegressor(ts_qs=[0.5, 0.90, 0.95]),
            get_abalone,
            parameter_grid,
            label=args.label,
            n_trials=args.n_trials,
            local_dir=args.local_dir,
            n_jobs=args.num_cores,
            time_budget_s=args.time_budget_s // len(total_budgets),
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def select_experiment(which: str) -> Callable[..., pd.DataFrame]:
    return dict(
        baseline=baseline,
        quantile_lin_comb=quantile_lin_comb,
        dp_rmse=dp_rmse,
        dp_rmse_ts=dp_rmse_ts,
        dp_rmse_ts_grid=dp_rmse_ts_grid,
        dp_quantile_ts=dp_quantile_ts,
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

import argparse
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold

import dpgbdt


def sklearn_grid(
    regressor,
    data_provider: Callable[[], dict[str, Any]],
    parameter_grid: dict[str, Any],
    n_jobs: Optional[int] = None,
    cv=None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    if cv is None:
        cv = RepeatedKFold(n_splits=5, n_repeats=2)
    sklearn_search = GridSearchCV(
        regressor,
        parameter_grid,
        n_jobs=n_jobs,
        scoring="neg_root_mean_squared_error",
        cv=cv,
    )
    sklearn_search.fit(**data_provider())
    df = pd.DataFrame(sklearn_search.cv_results_)
    return df


def get_abalone() -> Tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Parse the abalone dataset and return parameters suitable for
    `fit`.

    Returns:
      dict[str, Any]: An assignment to `fit`'s arguments.
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
    args = dict(y=data.rings.values.astype(float))
    del data["rings"]
    args["X"] = data.values.astype(float)
    args["cat_idx"] = [0]  # Sex
    args["num_idx"] = list(range(1, args["X"].shape[1]))  # Other attributes
    args["grid_lower_bounds"] = np.array(args["X"].shape[1] * [0.0])
    args["grid_upper_bounds"] = np.array([2.0, 1.0, 1.0, 1.5, 3.0, 2.0, 1.0, 1.5])
    args["grid_step_sizes"] = np.array([1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    return args


def get_wine() -> dict[str, Any]:
    """Parse the wine dataset.

    Returns:
      dict[str, Any]: An assignment to `fit`'s arguments.
    """
    data = pd.read_csv(
        "./datasets/real/winequality-red.csv",
        names=[
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
            "quality",
        ],
    )
    args = dict(y=data.quality.values.astype(float))
    del data["quality"]
    args["X"] = data.values.astype(float)
    args["cat_idx"] = []
    args["num_idx"] = list(range(1, args["X"].shape[1]))
    args["grid_lower_bounds"] = np.array(
        [4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0, 0.99, 2.5, 0.0, 7.0]
    )
    args["grid_upper_bounds"] = np.array(
        [16.0, 2.0, 1.0, 15.0, 1.0, 80.0, 300.0, 1.0, 5.0, 2, 18]
    )
    args["grid_step_sizes"] = np.array(
        [0.1, 0.01, 0.01, 0.1, 0.001, 1.0, 1.0, 0.0001, 0.01, 0.01, 0.1]
    )
    return args


def get_concrete() -> dict[str, Any]:
    """Parse the concrete dataset.

    Returns:
      dict[str, Any]: An assignment to `fit`'s arguments.
    """
    data = pd.read_csv(
        "./datasets/real/Concrete_Data_Yeh.csv",
        names=[
            "cement",
            "blast furnace slag",
            "fly ash",
            "water",
            "superplasticizer",
            "coarse aggregate",
            "fine aggregate",
            "age",
            "compressive_strength",
        ],
    )
    args = dict(y=data.age.values.astype(float))
    del data["quality"]
    args["X"] = data.values.astype(float)
    args["cat_idx"] = []
    args["num_idx"] = list(range(1, args["X"].shape[1]))
    args["grid_lower_bounds"] = np.array(args["X"].shape[1] * [0.0])
    args["grid_upper_bounds"] = np.array(
        [1000.0, 500.0, 500.0, 500.0, 100.0, 1500, 1500.0, 365.0]
    )
    args["grid_step_sizes"] = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0])
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform grid search over the hyperparameter space."
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


def abalone_parameter_grid_20221109():
    parameter_grid = dict(
        learning_rate=[0.01, 0.1, 1.0],
        max_depth=[1, 2, 6],
        # 20.0 is the max. difference between any target value and the
        # average target value
        l2_threshold=np.linspace(0.2, 20.0, 12),
        l2_lambda=[0.1],
        n_trees_to_accept=[1, 2, 3, 5, 10, 20, 50],
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


def abalone_parameter_dense_grid_20221107():
    parameter_grid = dict(
        learning_rate=[0.01, 0.1, 1.0],
        max_depth=[1, 2, 3, 4, 5, 6],
        # 20.0 is the max. difference between any target value and the
        # average target value
        l2_threshold=np.linspace(0.5, 20.0, 20),
        l2_lambda=np.linspace(0.1, 1.0, 10),
        n_trees_to_accept=[1, 2, 3, 5, 8, 10, 20, 50],
    )
    return parameter_grid


def wine_parameter_grid_20221121():
    parameter_grid = dict(
        learning_rate=[0.01, 0.1, 1.0],
        max_depth=[1, 5, 10],
        # 3.0 is the max. abs. difference between any target value and
        # the average target value
        l2_threshold=[0.03, 0.3, 3.0],
        l2_lambda=[0.01, 0.05, 0.1, 0.5, 1.0],
        n_trees_to_accept=[1, 5, 10, 20, 50, 100],
    )
    return parameter_grid


def baseline_template(
    args,
    grid: dict[str, Any],
    data_provider: Callable[[], dict[str, Any]] = None,
) -> pd.DataFrame:
    if data_provider is None:
        data_provider = get_abalone
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
            data_provider,
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


def baseline_dense_grid_20221107(args) -> pd.DataFrame:
    return baseline_template(args, abalone_parameter_dense_grid_20221107())


def baseline_grid_20221109(args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221109()
    grid["l2_lambda"] = [0.01, 0.1, 1.0]  # in contrast to just [0.1]
    return baseline_template(args, grid)


def dp_rmse_ts_template(
    args,
    grid: dict[str, Any],
    data_provider: Callable[[], dict[str, Any]] = None,
) -> pd.DataFrame:
    if data_provider is None:
        data_provider = get_abalone
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["tree_scorer"] = ["dp_rmse"]

        df = sklearn_grid(
            dpgbdt.DPGBDTRegressor(),
            data_provider,
            parameter_grid,
            n_jobs=args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def meta_template(
    cli_args,
    grid: dict[str, Any],
    fit_args: dict[str, Any],
) -> pd.DataFrame:
    dfs = []
    total_budgets = cli_args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid.copy()
        parameter_grid["privacy_budget"] = [total_budget]

        df = sklearn_grid(
            dpgbdt.DPGBDTRegressor(),
            data_provider=lambda: fit_args,
            parameter_grid=parameter_grid,
            n_jobs=cli_args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def bun_steinke_template(
    cli_args,
    grid: dict[str, Any],
    fit_args: dict[str, Any],
):
    grid = {**grid, **dict(tree_scorer=["bun_steinke"])}
    return meta_template(cli_args, grid, fit_args)


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


def dp_rmse_ts_grid_20221109(args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221109()
    grid["ensemble_rejector_budget_split"] = [0.01, 0.1, 0.2, 0.4, 0.6, 0.9]
    grid["dp_argmax_privacy_budget"] = [1e-5, 1e-4, 1e-3, 1e-2]
    grid["dp_argmax_stopping_prob"] = [0.001, 0.01, 0.1, 0.2, 0.4]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_gamma"] = [2]
    return dp_rmse_ts_template(args, grid)


def dp_quantile_ts_template(
    args,
    grid: dict[str, Any],
    ts_qs: list[float],
    data_provider: Callable[[], dict[str, Any]] = None,
) -> pd.DataFrame:
    if data_provider is None:
        data_provider = get_abalone
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["tree_scorer"] = ["dp_quantile"]

        df = sklearn_grid(
            dpgbdt.DPGBDTRegressor(ts_qs=ts_qs),
            data_provider,
            parameter_grid,
            n_jobs=args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def dp_quantile_ts_grid(args) -> pd.DataFrame:
    grid = abalone_parameter_grid()
    grid["ensemble_rejector_budget_split"] = [0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_shift"] = [0.0]
    grid["ts_scale"] = [0.79]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    return dp_quantile_ts_template(args, grid, ts_qs=[0.5, 0.90, 0.95])


def dp_quantile_ts_grid_20221107(args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221107()
    grid["ensemble_rejector_budget_split"] = [0.2, 0.4, 0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.0001, 0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.01, 0.1, 0.2, 0.4]
    grid["ts_shift"] = [0.0]
    grid["ts_scale"] = [0.5, 0.79, 1]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    return dp_quantile_ts_template(args, grid, ts_qs=[0.5, 0.90, 0.95])


def dp_quantile_ts_grid_20221109(args) -> pd.DataFrame:
    dfs = []
    for ts_qs in [[0.5, 0.90, 0.95], [0.5, 0.80, 0.95]]:
        grid = abalone_parameter_grid_20221109()
        grid["ensemble_rejector_budget_split"] = [0.01, 0.1, 0.2, 0.4, 0.6, 0.9]
        grid["dp_argmax_privacy_budget"] = [1e-5, 1e-4, 1e-3, 1e-2]
        grid["dp_argmax_stopping_prob"] = [0.001, 0.01, 0.1, 0.2, 0.4]
        grid["ts_shift"] = [0.0]
        grid["ts_scale"] = [0.5, 0.79, 1]
        grid["ts_upper_bound"] = grid["l2_threshold"]
        dfs.append(dp_quantile_ts_template(args, grid, ts_qs=ts_qs))
    return pd.concat(dfs)


def abalone_bun_steinke(cli_args) -> pd.DataFrame:
    grid = abalone_parameter_grid()
    grid["ensemble_rejector_budget_split"] = [0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_beta"] = np.logspace(-4.0, 0.0, 10)
    grid["ts_relaxation"] = [1e-6]
    return bun_steinke_template(cli_args, grid, get_abalone())


def wine_baseline_grid_20221121(args) -> pd.DataFrame:
    grid = wine_parameter_grid_20221121()
    return baseline_template(args, grid, data_provider=get_wine)


def wine_dp_rmse_ts_grid_20221121(args) -> pd.DataFrame:
    grid = wine_parameter_grid_20221121()
    grid["ensemble_rejector_budget_split"] = [0.5, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_gamma"] = [2]
    return dp_rmse_ts_template(args, grid, data_provider=get_wine)


def wine_dp_quantile_ts_grid_20221121(args) -> pd.DataFrame:
    grid = wine_parameter_grid_20221121()
    grid["ensemble_rejector_budget_split"] = [0.5, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_shift"] = [0.0]
    grid["ts_scale"] = [0.79]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    return dp_quantile_ts_template(
        args, grid, ts_qs=[0.5, 0.90, 0.95], data_provider=get_wine
    )


def select_experiment(which: str) -> Callable[..., pd.DataFrame]:
    return dict(
        baseline_grid=baseline_grid,
        baseline_grid_20221107=baseline_grid_20221107,
        baseline_grid_20221109=baseline_grid_20221109,
        baseline_dense_grid=baseline_dense_grid,
        baseline_dense_grid_20221107=baseline_dense_grid_20221107,
        dp_rmse_ts_grid=dp_rmse_ts_grid,
        dp_rmse_ts_grid_20221107=dp_rmse_ts_grid_20221107,
        dp_rmse_ts_grid_20221109=dp_rmse_ts_grid_20221109,
        dp_quantile_ts_grid=dp_quantile_ts_grid,
        dp_quantile_ts_grid_20221107=dp_quantile_ts_grid_20221107,
        dp_quantile_ts_grid_20221109=dp_quantile_ts_grid_20221109,
        wine_baseline_grid_20221121=wine_baseline_grid_20221121,
        wine_dp_rmse_ts_grid_20221121=wine_dp_rmse_ts_grid_20221121,
        wine_dp_quantile_ts_grid_20221121=wine_dp_quantile_ts_grid_20221121,
        abalone_bun_steinke=abalone_bun_steinke,
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

import argparse
import math
import traceback
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

import dpgbdt
import pyestimator
from datasets.real import data_reader


def manual_grid(
    fit_args: dict[str, Any],
    parameter_grid: dict[str, Any],
    n_repetitions: int,
    cli_args,
    config_processor: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    n_jobs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """Traverse the parameter grid manually to control the propagation
    of random seeds.

    Returns:
        pd.DataFrame: A DataFrame containing the configurations and
        their corresponding scores. Notice that, in contrast to
        sklearn_grid, this DataFrame contains a new line for every
        configuration split combination (tall format). The other
        DataFrame contains only one line per configuration and holds
        all split scores in its columns (wide format).
    """
    if rng is None:
        rng = np.random.default_rng()
    if config_processor is None:
        config_processor = lambda x: x
    seeds = rng.integers(low=0, high=2**30 - 1, size=n_repetitions)

    scores_df = pd.DataFrame()
    for i, configs_chunk in enumerate(
        _configs_chunks(model_selection.ParameterGrid(parameter_grid))
    ):
        seeded_configs_chunk = product(configs_chunk, seeds)
        scores_chunk = joblib.Parallel(n_jobs=n_jobs, verbose=50)(
            joblib.delayed(_manual_worker_wrapper(fit_args))(config, seed)
            for (config, seed) in seeded_configs_chunk
        )
        scores_df = pd.concat([scores_df, pd.DataFrame(scores_chunk)])
        _write_intermediate_scores_to_disk(
            df=scores_df, suffix=f".chunk={i}", cli_args=cli_args
        )
    return scores_df


def _configs_chunks(
    configs: model_selection.ParameterGrid, chunk_size: int = 1024
) -> Iterable[list[dict[str, Any]]]:
    configs_list = list(configs)
    n = len(configs_list)
    return (
        configs_list[chunk_size * i : chunk_size * (i + 1)]
        for i in range(math.ceil(n / chunk_size))
    )


def _write_intermediate_scores_to_disk(df: pd.DataFrame, suffix: str, cli_args):
    p = cli_args.intermediate_results_dir / f"{cli_args.csvfilename.stem}{suffix}.csv"
    df.to_csv(p, mode="w")


def _manual_worker_wrapper(fit_data: dict[str, Any]):
    _fit_data = fit_data.copy()
    X = _fit_data.pop("X")
    y = _fit_data.pop("y")

    def _manual_worker(config: dict[str, Any], seed: int) -> dict[str, Any]:
        """Create a DPGBDT ensemble and split fit_data, governed by `seed`,
        and train the ensemble on that split.

        Returns:
            dict[str, Any]: The configuration used, prepended by "param_",
            the seed and the resulting rMSE test score.
        """
        config_incl_seed = dict(**config, seed=seed)
        config_check = check_config(config_incl_seed)
        if config_check:
            raise ValueError(
                f"Some of these parameters {config_check} are missing "
                f"in config {config_incl_seed}"
            )

        regressor = dpgbdt.DPGBDTRegressor(**config_incl_seed)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        try:
            regressor.fit(X=X_train, y=y_train, **_fit_data)
            y_pred = regressor.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
        except Exception:
            traceback.print_exc()
            rmse = float("nan")
            r2 = float("nan")
        params = {f"param_{key}": val for (key, val) in config_incl_seed.items()}
        return dict(**params, test_score=rmse, test_r2_score=r2)

    return _manual_worker


def check_config(config: dict[str, Any]) -> list[str]:
    """Check for missing parameters to `DPGBDTRegressor`.

    Returns:
        list[str]: If `[]`, no parameter is missing. Otherwise at least
        one of the returned parameters is missing (but not necessarily
        all of them).
    """
    keys = config.keys()
    common_args = [
        "seed",
        "privacy_budget",
        "training_variant",
        "learning_rate",
        "n_trees_to_accept",
        "max_depth",
        "l2_threshold",
        "l2_lambda",
    ]
    common_args_present = all([arg in keys for arg in common_args])
    if common_args_present:
        if config["training_variant"] == "vanilla":
            return []
        elif config["training_variant"] == "dp_argmax_scoring":
            argmax_scoring_args = [
                "ensemble_rejector_budget_split",
                "tree_scorer",
                "dp_argmax_privacy_budget",
                "dp_argmax_stopping_prob",
            ]
            argmax_scoring_args_present = all(
                [arg in keys for arg in argmax_scoring_args]
            )
            if argmax_scoring_args_present:
                t = type(config["tree_scorer"])
                scorer_args = []
                if t in [pyestimator.PyDPrMSEScorer, pyestimator.PyDPrMSEScorer2]:
                    scorer_args = ["ts_upper_bound", "ts_gamma"]
                elif t is pyestimator.PyDPQuantileScorer:
                    scorer_args = ["ts_shift", "ts_scale", "ts_qs", "ts_upper_bound"]
                elif t is pyestimator.PyBunSteinkeScorer:
                    scorer_args = ["ts_upper_bound", "ts_beta", "ts_relaxation"]
                elif t is pyestimator.PyPrivacyBucketScorer:
                    scorer_args = ["ts_upper_bound", "ts_beta", "ts_relaxation"]
                elif t is str:
                    # assume that the conversion from str to real
                    # parameters works correctly
                    return []
                else:
                    raise ValueError(f"Exhausted tree scorer options: {t}")
                if all([arg in keys for arg in scorer_args]):
                    return []
                else:
                    return scorer_args
            else:
                return argmax_scoring_args
        else:
            raise ValueError(f"Unknown training variant {config['training_variant']}")
    else:
        return common_args


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
    data_provider: Optional[Callable[[], dict[str, Any]]] = None,
) -> pd.DataFrame:
    if data_provider is None:
        data_provider = data_reader.abalone_fit_arguments
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
        df = manual_grid(
            fit_args=data_provider(),
            parameter_grid=parameter_grid,
            cli_args=args,
            n_repetitions=50,
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
    data_provider: Optional[Callable[[], dict[str, Any]]] = None,
) -> pd.DataFrame:
    if data_provider is None:
        data_provider = data_reader.abalone_fit_arguments
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid
        grid["training_variant"] = ["dp_argmax_scoring"]
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["tree_scorer"] = ["dp_rmse"]

        df = manual_grid(
            fit_args=data_provider(),
            parameter_grid=parameter_grid,
            cli_args=args,
            n_repetitions=50,
            n_jobs=args.num_cores,
        )
        dfs.append(df)
    df = pd.concat(dfs)
    return df


def meta_template(
    cli_args,
    grid: dict[str, Any],
    **kwargs,
) -> pd.DataFrame:
    cli_args.intermediate_results_dir.mkdir(parents=True, exist_ok=True)
    scores_df = pd.DataFrame()
    total_budgets = cli_args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid.copy()
        parameter_grid["privacy_budget"] = [total_budget]
        df = manual_grid(
            parameter_grid=parameter_grid,
            cli_args=cli_args,
            n_jobs=cli_args.num_cores,
            **kwargs,
        )
        scores_df = pd.concat([scores_df, df])
        _write_intermediate_scores_to_disk(
            scores_df, suffix=f".eps={total_budget}", cli_args=cli_args
        )
    return scores_df


def bun_steinke_template(
    cli_args,
    grid: dict[str, Any],
    fit_args: dict[str, Any],
):
    grid["tree_scorer"] = ["bun_steinke"]
    grid["training_variant"] = ["dp_argmax_scoring"]
    return meta_template(cli_args, grid, fig_args=fit_args)


def privacy_bucket_template(
    cli_args,
    grid: dict[str, Any],
    fit_args: dict[str, Any],
):
    grid["tree_scorer"] = ["privacy_buckets"]
    grid["training_variant"] = ["dp_argmax_scoring"]
    return meta_template(cli_args, grid, fig_args=fit_args)


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
    data_provider: Optional[Callable[[], dict[str, Any]]] = None,
) -> pd.DataFrame:
    if data_provider is None:
        data_provider = data_reader.abalone_fit_arguments
    dfs = []
    total_budgets = args.privacy_budgets
    for total_budget in total_budgets:
        parameter_grid = grid
        parameter_grid["training_variant"] = ["dp_argmax_scoring"]
        parameter_grid["privacy_budget"] = [total_budget]
        parameter_grid["tree_scorer"] = ["dp_quantile"]
        parameter_grid["ts_qs"] = [ts_qs]

        df = manual_grid(
            fit_args=data_provider(),
            parameter_grid=parameter_grid,
            cli_args=args,
            n_repetitions=50,
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
    grid["ts_beta"] = [729 * 1e-6]
    grid["ts_relaxation"] = [1e-6]
    return bun_steinke_template(cli_args, grid, data_reader.abalone_fit_arguments())


def abalone_bun_steinke_20221107(cli_args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221107()
    grid["ensemble_rejector_budget_split"] = [0.2, 0.4, 0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.0001, 0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.01, 0.1, 0.2, 0.4]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_beta"] = [729 * 1e-6]
    grid["ts_relaxation"] = [1e-6]
    return bun_steinke_template(cli_args, grid, data_reader.abalone_fit_arguments())


def abalone_bun_steinke_20230321(cli_args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221107()
    grid["ensemble_rejector_budget_split"] = [0.2, 0.4, 0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.0001, 0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.01, 0.1, 0.2, 0.4]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_beta"] = [818 * 1e-6]
    grid["ts_relaxation"] = [1e-6]
    return bun_steinke_template(cli_args, grid, data_reader.abalone_fit_arguments())


def abalone_privacy_buckets(cli_args) -> pd.DataFrame:
    grid = abalone_parameter_grid()
    grid["ensemble_rejector_budget_split"] = [0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_beta"] = [729 * 1e-6]
    grid["ts_relaxation"] = [1e-6]
    grid["ts_coefficients"] = [[0.0, 1.25, -0.0361, 0.0, 0.0, -2.10, 0.0, 0.0, 0.0]]  # type: ignore
    return privacy_bucket_template(cli_args, grid, data_reader.abalone_fit_arguments())


def abalone_privacy_buckets_20221107(cli_args) -> pd.DataFrame:
    grid = abalone_parameter_grid_20221107()
    grid["ensemble_rejector_budget_split"] = [0.2, 0.4, 0.6, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.0001, 0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.01, 0.1, 0.2, 0.4]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_beta"] = [729 * 1e-6]
    grid["ts_relaxation"] = [1e-6]
    grid["ts_coefficients"] = [[0.0, 1.25, -0.0361, 0.0, 0.0, -2.10, 0.0, 0.0, 0.0]]  # type: ignore
    return privacy_bucket_template(cli_args, grid, data_reader.abalone_fit_arguments())


def wine_baseline_grid_20221121(args) -> pd.DataFrame:
    grid = wine_parameter_grid_20221121()
    return baseline_template(args, grid, data_provider=data_reader.wine_fit_arguments)


def wine_dp_rmse_ts_grid_20221121(args) -> pd.DataFrame:
    grid = wine_parameter_grid_20221121()
    grid["ensemble_rejector_budget_split"] = [0.5, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    grid["ts_gamma"] = [2]
    return dp_rmse_ts_template(args, grid, data_provider=data_reader.wine_fit_arguments)


def wine_dp_quantile_ts_grid_20221121(args) -> pd.DataFrame:
    grid = wine_parameter_grid_20221121()
    grid["ensemble_rejector_budget_split"] = [0.5, 0.75, 0.9]
    grid["dp_argmax_privacy_budget"] = [0.001, 0.01]
    grid["dp_argmax_stopping_prob"] = [0.1, 0.2]
    grid["ts_shift"] = [0.0]
    grid["ts_scale"] = [0.79]
    grid["ts_upper_bound"] = grid["l2_threshold"]
    return dp_quantile_ts_template(
        args,
        grid,
        ts_qs=[0.5, 0.90, 0.95],
        data_provider=data_reader.wine_fit_arguments,
    )


def metro_baseline_grid_20230425(args) -> pd.DataFrame:
    params = dict(
        learning_rate=[0.1],
        max_depth=[1, 5, 10],
        # 4500 is roughly the value of
        #     | traffic_volume.mean() - traffic_volume.max() |
        l2_threshold=np.linspace(10.0, 4500.0, 10),
        l2_lambda=[0.1, 1.0, 10.0],
        n_trees_to_accept=[5, 10, 50, 100],
        training_variant=["vanilla"],
    )
    return meta_template(
        args,
        params,
        fit_args=data_reader.metro_fit_arguments(),
        n_repetitions=5,
    )


def metro_baseline_grid_20230426(args) -> pd.DataFrame:
    params = dict(
        learning_rate=[0.1],
        max_depth=[1, 5],
        # 4500 is roughly the value of
        #     | traffic_volume.mean() - traffic_volume.max() |
        l2_threshold=np.linspace(10.0, 4500.0, 10),
        l2_lambda=[1.0, 10.0, 100],
        n_trees_to_accept=[10, 20],
        training_variant=["vanilla"],
    )
    return meta_template(
        args,
        params,
        fit_args=data_reader.metro_fit_arguments(),
        n_repetitions=5,
    )


def metro_bunsteinke_grid_20230425(args) -> pd.DataFrame:
    params = dict(
        learning_rate=[0.1],
        max_depth=[1, 6],
        # 4500 is roughly the value of
        #     | traffic_volume.mean() - traffic_volume.max() |
        l2_threshold=np.linspace(10.0, 4500.0, 10),
        l2_lambda=[0.1, 1.0, 5.0, 10.0],
        n_trees_to_accept=[5, 10, 20],
        tree_scorer=["bun_steinke"],
        training_variant=["dp_argmax_scoring"],
        ensemble_rejector_budget_split=[0.2, 0.5, 0.8],
        dp_argmax_privacy_budget=[0.001, 0.01],
        dp_argmax_stopping_prob=[0.01, 0.1],
        ts_beta=[729 * 1e-6],
        ts_relaxation=[1e-6],
    )
    # coupling of ts_upper_bound to l2_threshold
    cp = lambda config: {
        "ts_upper_bound": config["l2_threshold"],
        **config,
    }  # good idea?
    return meta_template(
        args,
        params,
        fit_args=data_reader.metro_fit_arguments(),
        n_repetitions=5,
        config_processor=cp,
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
        abalone_bun_steinke_20221107=abalone_bun_steinke_20221107,
        abalone_bun_steinke_20230321=abalone_bun_steinke_20230321,
        abalone_privacy_buckets=abalone_privacy_buckets,
        abalone_privacy_buckets_20221107=abalone_privacy_buckets_20221107,
        metro_baseline_grid_20230425=metro_baseline_grid_20230425,
        metro_bunsteinke_grid_20230425=metro_bunsteinke_grid_20230425,
        metro_baseline_grid_20230426=metro_baseline_grid_20230426,
    )[which]


if __name__ == "__main__":
    args = parse_args()
    if args.label is None:
        args.label = args.experiment
    if args.local_dir is None:
        args.local_dir = f"~/share/dp-gbdt-evaluation/"
    if args.csvfilename is None:
        args.csvfilename = Path(args.local_dir).expanduser() / f"{args.label}.csv"
    args.intermediate_results_dir = (
        Path(args.local_dir).expanduser() / "intermediate_results"
    )
    experiment = select_experiment(args.experiment)
    df = experiment(args)
    df.to_csv(args.csvfilename)

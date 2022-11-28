import contextlib
import os
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import dpgbdt
from example_main import abalone_fit_arguments
from stdout_redirector import stdout_redirector

COL_NAME_MAPPING: dict[str, str] = dict(
    param_learning_rate="learning_rate",
    param_max_depth="max_depth",
    param_l2_threshold="l2_threshold",
    param_l2_lambda="l2_lambda",
    param_n_trees_to_accept="n_trees_to_accept",
    param_privacy_budget="privacy_budget",
    param_ensemble_rejector_budget_split="ensemble_rejector_budget_split",
    param_tree_scorer="tree_scorer",
    param_dp_argmax_privacy_budget="dp_argmax_privacy_budget",
    param_dp_argmax_stopping_prob="dp_argmax_stopping_prob",
    param_ts_upper_bound="ts_upper_bound",
    param_ts_gamma="ts_gamma",
    param_ts_shift="ts_shift",
    param_ts_scale="ts_scale",
    param_training_variant="training_variant",
)


def params_from_series(series: pd.Series) -> dict[str, Any]:
    extracted_hyperparams = {}
    for (col_name, hyperparam) in COL_NAME_MAPPING.items():
        if col_name in series:
            extracted_hyperparams[hyperparam] = series[col_name]
    return extracted_hyperparams


def param_seq_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [params_from_series(df.iloc[i]) for i in range(len(df))]


def extend_params(params: dict[str, Any], **kwargs) -> dict[str, Any]:
    return {**params, **kwargs}


def _rmse(x, y) -> float:
    return np.sqrt(np.mean(np.square(x - y)))


def baseline_sanity_check():
    fit_args = abalone_fit_arguments()
    X = fit_args.pop("X")
    y = fit_args.pop("y")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df = pd.read_csv("baseline.csv")
    paramss = param_seq_from_df(
        df[(df["mean_test_score"] > -2.6) & (df["param_privacy_budget"] == 1.0)].sample(
            30
        )
    )

    for (i, params) in enumerate(paramss):
        print(f"{10 * '#'} Trial {i} {10 * '#'}")
        extended_params = extend_params(
            params,
            training_variant="vanilla",
            tree_scorer=None,
        )

        print(f"Parameters: {params}")
        estimator = dpgbdt.DPGBDTRegressor(**extended_params)
        estimator.fit(X_train, y_train, **fit_args)
        print(f"fitted estimator: {estimator}")
        score = _rmse(estimator.predict(X_test), y_test)
        if score <= 2.6:
            print(f"{20 * '>'} SUCCESS: score of {score} {20 * '<'}")
        else:
            print(f"score of just {score}")


def single_configuration(
    row: pd.Series,
    additional_parameters: dict[str, Any],
    fit_args: dict[str, Any],
    logfilename: str = None,
) -> dpgbdt.DPGBDTRegressor:
    """Relaunch a single configuration of the DP-GBDT regressor,
    obtaining parameters from a single row of the result DataFrame.

    Args:
        row (pd.row): the row containing the parameters of the
            configuration.
        additional_parameters (dict[str, Any]): further parameters not
            contained in the row
        fit_args (dict[str, Any]): the arguments required by the
            regressors's fit method
        logfilename (str, optional): If provided, create this logfile
            and redirect stdout to it. Otherwise don't redirect stdout.
            Defaults to None.
    """
    _fit_args = fit_args.copy()
    X = _fit_args.pop("X")
    y = _fit_args.pop("y")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    params = params_from_series(row)
    params = {**params, **additional_parameters}

    if logfilename is not None:
        maybe_redirecting = contextlib.ExitStack()
        logfile = maybe_redirecting.enter_context(open(logfilename, "wb"))
        maybe_redirecting.enter_context(stdout_redirector(logfile))
    else:
        maybe_redirecting = contextlib.nullcontext()
    # mind that when actually redirecting, python's stdout and the
    # extension's stdout might not be in correct order relative to
    # another (but at least within their own stdout they should be
    # consistent)
    with maybe_redirecting:
        print(f"Parameters: {params}")
        estimator = dpgbdt.DPGBDTRegressor(**params)
        estimator.fit(X_train, y_train, **_fit_args)
        print(f"fitted estimator: {estimator}")
        score = _rmse(estimator.predict(X_test), y_test)
        print(f"score: {score}")
    return estimator


def multiple_configurations(
    df: pd.DataFrame,
    indices: list[int],
    additional_parameters: dict[str, Any],
    fit_args: dict[str, Any],
    logfilename_template: str,
    zipfilename: Optional[str] = None,
):
    indices_and_logfiles = [
        (index, logfilename_template.format(index=index)) for index in indices
    ]
    estimators = []
    for (index, logfile) in indices_and_logfiles:
        estimator = single_configuration(
            df.loc[index],
            additional_parameters=additional_parameters,
            fit_args=fit_args,
            logfilename=logfile,
        )
        estimators.append(estimator)
    if zipfilename is not None:
        with zipfile.ZipFile(
            zipfilename, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zfile:
            for (_, logfile) in indices_and_logfiles:
                # print(f"Zipping and removing logfile {logfile}")
                zfile.write(logfile)
                os.remove(logfile)
    return estimators


def all_configurations(
    df: pd.DataFrame,
    additional_parameters: dict[str, Any],
    fit_args: dict[str, Any],
    logfilename_template: str,
    zipfilename: Optional[str] = None,
):
    return multiple_configurations(
        df=df,
        indices=df.index.values,
        additional_parameters=additional_parameters,
        fit_args=fit_args,
        logfilename_template=logfilename_template,
        zipfilename=zipfilename,
    )


def baseline(
    series: pd.Series,
    fit_args: dict[str, Any] = None,
    verbosity: str = "debug",
    logfilename: str = None,
):
    if fit_args is None:
        fit_args = abalone_fit_arguments()
    additional_params = dict(
        training_variant="vanilla",
        tree_scorer=None,
        verbosity=verbosity,
    )
    single_configuration(
        row=series,
        fit_args=fit_args,
        additional_parameters=additional_params,
        logfilename=logfilename,
    )


def dp_rmse(
    series: pd.Series,
    fit_args: dict[str, Any],
    verbosity: str = "debug",
    logfilename: str = None,
):
    additional_params = dict(
        training_variant="dp_argmax_scoring",
        tree_scorer="dp_rmse",
        verbosity=verbosity,
    )
    single_configuration(
        row=series,
        fit_args=fit_args,
        additional_parameters=additional_params,
        logfilename=logfilename,
    )


def dp_quantile():
    return None


def best_scores(df: pd.DataFrame) -> pd.DataFrame:
    df2 = (df[df["rank_test_score"] == 1]).copy()
    df2["mean_test_score"] = -1 * df2["mean_test_score"]
    try:
        df2 = df2[
            [
                "mean_test_score",
                "std_test_score",
                "param_privacy_budget",
                "param_learning_rate",
                "param_max_depth",
                "param_l2_threshold",
                "param_l2_lambda",
                "param_n_trees_to_accept",
                "param_dp_argmax_privacy_budget",
                "param_dp_argmax_stopping_prob",
                "param_ensemble_rejector_budget_split",
                "param_ts_upper_bound",
                "rank_test_score",
            ]
        ]
    except KeyError:
        df2 = df2[
            [
                "mean_test_score",
                "std_test_score",
                "param_privacy_budget",
                "param_learning_rate",
                "param_max_depth",
                "param_l2_threshold",
                "param_l2_lambda",
                "param_n_trees_to_accept",
                "rank_test_score",
            ]
        ]
    return df2


def log_best_abalone_configurations():
    additional_params = dict(
        training_variant="dp_argmax_scoring",
        tree_scorer="dp_rmse",
        verbosity="debug",
    )
    experiments = [
        "baseline_dense-gridspace_20221107_feature-grid.csv",
        "baseline_gridspace_20221107_feature-grid.csv",
        "dp_rmse_ts_gridspace_feature-grid.csv",
        "dp_rmse_ts_gridspace_20221107_feature-grid.csv",
        "dp_quantile_ts_gridspace_feature-grid.csv",
        "dp_quantile_ts_gridspace_20221107_feature-grid.csv",
    ]
    for experiment in experiments:
        p = Path(experiment)
        df = pd.read_csv(experiment)
        df = best_scores(df)
        all_configurations(
            df=df,
            additional_parameters=additional_params,
            fit_args=abalone_fit_arguments(),
            logfilename_template=f"{p.stem}" + ".{index}.log",
            zipfilename=f"{p.stem}.zip",
        )


def dp_rmse_score_variation():
    y = abalone_fit_arguments().pop("y")
    y_pred = np.full_like(y, y.mean())
    scores_dict = {}
    for eps in np.logspace(-3.0, 1.0, 30):
        rmse_list = []
        for i in range(100):
            rng = dpgbdt.make_rng(i)
            scorer = dpgbdt.make_tree_scorer(
                "dp_rmse", upper_bound=20.0, gamma=2.0, rng=rng
            )
            rmse = scorer.score_tree(eps, y, y_pred)
            rmse_list.append(rmse)
            scores_dict[f"eps={eps}"] = rmse_list
    df = pd.DataFrame(scores_dict)
    df.to_csv("rmse_variability.csv", index=False)


if __name__ == "__main__":
    # log_best_abalone_configurations()
    dp_rmse_score_variation()

import contextlib
import json
import os
import zipfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

import dpgbdt
from example_main import abalone_fit_arguments
from hyperparameter_tuning import check_config
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
    param_ts_beta="ts_beta",
    param_ts_relaxation="ts_relaxation",
    param_training_variant="training_variant",
    param_seed="seed",
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


def single_configuration(
    row: pd.Series,
    additional_parameters: dict[str, Any],
    fit_args: dict[str, Any],
    logfilename: Optional[str] = None,
    cv=None,
) -> Tuple[dpgbdt.DPGBDTRegressor, np.ndarray]:
    """Relaunch a single configuration of the DP-GBDT regressor,
    obtaining parameters from a single row of the result DataFrame.

    Args:
        row (pd.Series): the row containing the parameters of the
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

    params = params_from_series(row)
    params = {**params, **additional_parameters}
    config_check = check_config(params)
    if config_check:
        raise ValueError(
            f"Some of these parameters {config_check} are missing "
            f"in config {params}"
        )

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
        if cv is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=params["seed"]
            )
            estimator.fit(X_train, y_train, **_fit_args)
            print(f"fitted estimator: {estimator}")
            score = _rmse(estimator.predict(X_test), y_test)
            print(f"score: {score}")
            return (estimator, np.array(score))
        else:
            scores = cross_val_score(
                estimator,
                X,
                y,
                fit_params=_fit_args,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=32,
            )
            print(f"fitted estimator: {estimator}")
            print(f"scores: {scores}")
            return (estimator, scores)


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
        indices=df.index.values,  # type: ignore
        additional_parameters=additional_parameters,
        fit_args=fit_args,
        logfilename_template=logfilename_template,
        zipfilename=zipfilename,
    )


def baseline(
    series: pd.Series,
    fit_args: Optional[dict[str, Any]] = None,
    verbosity: str = "debug",
    logfilename: Optional[str] = None,
    cv=None,
) -> dpgbdt.DPGBDTRegressor:
    if fit_args is None:
        fit_args = abalone_fit_arguments()
    additional_params = dict(
        training_variant="vanilla",
        tree_scorer=None,
        verbosity=verbosity,
    )
    return single_configuration(
        row=series,
        fit_args=fit_args,
        additional_parameters=additional_params,
        logfilename=logfilename,
        cv=cv,
    )[0]


def dp_rmse(
    series: pd.Series,
    fit_args: dict[str, Any],
    verbosity: str = "debug",
    logfilename: Optional[str] = None,
    cv=None,
) -> dpgbdt.DPGBDTRegressor:
    additional_params = dict(
        training_variant="dp_argmax_scoring",
        tree_scorer="dp_rmse",
        verbosity=verbosity,
    )
    return single_configuration(
        row=series,
        fit_args=fit_args,
        additional_parameters=additional_params,
        logfilename=logfilename,
        cv=cv,
    )[0]


def dp_quantile():
    return None


def bun_steinke(
    series: pd.Series,
    fit_args: dict[str, Any],
    verbosity: str = "debug",
    logfilename: Optional[str] = None,
    cv=None,
) -> dpgbdt.DPGBDTRegressor:
    additional_params = dict(
        training_variant="dp_argmax_scoring",
        tree_scorer="bun_steinke",
        verbosity=verbosity,
    )
    return single_configuration(
        row=series,
        fit_args=fit_args,
        additional_parameters=additional_params,
        logfilename=logfilename,
        cv=cv,
    )[0]


def best_scores(df: pd.DataFrame) -> pd.DataFrame:
    if "rank_test_score" in df.columns:
        return df[df["rank_test_score"] == 1.0]
    else:
        # we have to calculate the ranks first
        all_params_but_seed = [
            col
            for col in df.columns
            if col.startswith("param_") and col != "param_seed"
        ]
        agg_df = (
            df.groupby(by=all_params_but_seed)
            .agg(
                mean_test_score=pd.NamedAgg(column="test_score", aggfunc="mean"),
            )
            .reset_index()
        )
        agg_df["rank_test_score"] = agg_df.groupby(by=["param_privacy_budget"])[
            "mean_test_score"
        ].rank(method="min")
        agg_df = agg_df[agg_df["rank_test_score"] == 1.0]
        merged = pd.merge(df, agg_df, on=all_params_but_seed, how="inner")
        return merged


def log_best_abalone_configurations(experiments: Optional[list[str]] = None):
    if experiments is None:
        experiments_path = Path(
            "~/share/dp-gbdt-evaluation/abalone_experiments.json"
        ).expanduser()
        with open(experiments_path, "r") as f:
            experiments = json.load(f)
    for experiment in experiments:  # type: ignore
        p = Path(experiment).expanduser()
        df = pd.read_csv(experiment)
        df = best_scores(df)
        additional_params = dict(
            verbosity="debug",
        )
        all_configurations(
            df=df,
            additional_parameters=additional_params,
            fit_args=abalone_fit_arguments(),
            # needs to be a (format) string, not a Path
            logfilename_template=f"{p.parent}/{p.stem}" + ".{index}.log",
            zipfilename=str(p.with_suffix(".zip")),
        )


def log_best_wine_configurations():
    pass


def dp_rmse_score_variation():
    """How variable is the DP-RMSE-scoring with respect to different
    privacy budgets?
    """
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
            scores_dict[eps] = rmse_list
    df = pd.DataFrame(scores_dict)
    df = (
        pd.melt(df, var_name="eps", value_name="dp-rmse")
        .groupby(["eps"], as_index=False)
        .agg(
            mean_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="mean"),
            std_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="std"),
        )
        .rename(
            {"std_dp_rmse": "STD(dp-rmse)", "mean_dp_rmse": "MEAN(dp-rmse)"}, axis=1
        )
    )
    df = df.assign(rmse=np.sqrt(((y - y_pred) ** 2).mean()))
    df.to_csv("rmse_variability.csv", index=False)


def dp_rmse2_score_variation():
    """How variable is the DP-RMSE-scoring with respect to different
    privacy budgets and different betas?
    """
    y = abalone_fit_arguments().pop("y")
    y_pred = np.full_like(y, y.mean())
    rmse = np.sqrt(((y - y_pred) ** 2).mean())
    records = []
    for beta in np.logspace(-5, 1, 30):
        for eps in np.logspace(-3.0, 1.0, 30):
            for i in range(100):
                rng = dpgbdt.make_rng(i)
                scorer = dpgbdt.make_tree_scorer(
                    "dp_rmse2",
                    beta=dpgbdt.make_beta("constant_beta", beta=beta),
                    upper_bound=20.0,
                    gamma=2.0,
                    rng=rng,
                )
                dp_rmse = scorer.score_tree(eps, y, y_pred)
                records.append((beta, eps, rmse, dp_rmse))
    df = pd.DataFrame.from_records(records, columns=["beta", "eps", "rmse", "dp-rmse"])
    df = (
        df.groupby(["beta", "eps"])
        .agg(
            rmse=pd.NamedAgg(column="rmse", aggfunc="first"),
            mean_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="mean"),
            std_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="std"),
        )
        .reset_index()
        .rename(
            {"mean_dp_rmse": "MEAN(dp-rmse)", "std_dp_rmse": "STD(dp-rmse)"},
            axis=1,
        )
    )
    df["STD(dp-rmse)/rmse"] = df["STD(dp-rmse)"] / df["rmse"]
    df.to_csv("dp_rmse2_score_variation.csv", index=False)


def dp_rmse_score_variation_dataset_size_vs_privacy_budget():
    """See whether it helps to trade in dataset size for privacy budget.
    In detail: Does it help to reduce the dataset size to x %, while
    multiplying the available budget by x?
    """
    np_rng = np.random.default_rng()
    records = []
    for fraction in np.linspace(0.1, 1.0, 10):
        y = abalone_fit_arguments().pop("y")
        y = np_rng.choice(y, size=int(fraction * len(y)), replace=False)
        y_pred = np.full_like(y, y.mean())
        rmse = np.sqrt(((y - y_pred) ** 2).mean())
        for eps in np.logspace(-3.0, 1.0, 30):
            for i in range(100):
                rng = dpgbdt.make_rng(i)
                scorer = dpgbdt.make_tree_scorer(
                    "dp_rmse", upper_bound=20.0, gamma=2.0, rng=rng
                )
                dp_rmse = scorer.score_tree((1 / fraction) * eps, y, y_pred)
                records.append((fraction, eps, rmse, dp_rmse))
    df = pd.DataFrame.from_records(
        records, columns=["fraction", "eps", "rmse", "dp-rmse"]
    )
    df = (
        df.groupby(["eps", "fraction"])
        .agg(
            rmse=pd.NamedAgg(column="rmse", aggfunc="first"),
            mean_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="mean"),
            std_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="std"),
        )
        .reset_index()
        .rename(
            {"std_dp_rmse": "STD(dp-rmse)", "mean_dp_rmse": "MEAN(dp-rmse)"}, axis=1
        )
    )
    df.to_csv("dp_rmse_score_variation_dataset_size_vs_privacy_budget.csv", index=False)


def dp_rmse_score_variation_bun_steinke():
    y = abalone_fit_arguments().pop("y")
    y_pred = np.full_like(y, y.mean())
    rmse = np.sqrt(((y - y_pred) ** 2).mean())
    records = []
    for relaxation in np.logspace(-6, -1, 6):
        for beta in np.logspace(-5, 1, 30):
            for eps in np.logspace(-3.0, 1.0, 30):
                for i in range(100):
                    rng = dpgbdt.make_rng(i)
                    scorer = dpgbdt.make_tree_scorer(
                        "bun_steinke",
                        upper_bound=20.0,
                        beta=beta,
                        relaxation=relaxation,
                        rng=rng,
                    )
                    dp_rmse = scorer.score_tree(eps, y, y_pred)
                    records.append((relaxation, beta, eps, rmse, dp_rmse))
    df = pd.DataFrame.from_records(
        records, columns=["relaxation", "beta", "eps", "rmse", "dp-rmse"]
    )
    df = (
        df.groupby(["relaxation", "beta", "eps"])
        .agg(
            rmse=pd.NamedAgg(column="rmse", aggfunc="first"),
            mean_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="mean"),
            std_dp_rmse=pd.NamedAgg(column="dp-rmse", aggfunc="std"),
        )
        .reset_index()
        .rename(
            {"mean_dp_rmse": "MEAN(dp-rmse)", "std_dp_rmse": "STD(dp-rmse)"},
            axis=1,
        )
    )
    df["STD(dp-rmse)/rmse"] = df["STD(dp-rmse)"] / df["rmse"]
    df.to_csv("dp_rmse_score_variation_bun_steinke.csv", index=False)


def repeat_crossvalidation_baseline(
    setting: pd.Series,
    additional_settings: dict[str, Any],
    n_repetitions: int,
    cv,
    n_jobs=32,
) -> pd.DataFrame:
    scores = []
    _setting = setting.copy()
    for (param, value) in additional_settings.items():
        _setting[param] = value
    scoress = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(baseline)(series=_setting, cv=cv) for _ in range(n_repetitions)
    )
    scores = [score for (_, scores) in scoress for score in scores]  # type: ignore
    df = pd.DataFrame(scores, columns=["rmse"])
    df["rmse"] = df["rmse"] * -1.0
    return df


def diff_logs(log1, log2, ignore_timestamp: bool = True) -> Optional[Tuple[str, str]]:
    """Compare two logs line by line, possibly ignoring the leading
    timestamps.

    Returns:
        Optional[Tuple[str, str]]: If `log1` and `log2` differ, return the
        first differing two lines (left from `log1`, right from `log2`).
        Otherwise return `None`.
    """
    with open(log1, "r") as file1:
        with open(log2, "r") as file2:
            lines1 = file1.readlines()
            lines2 = file2.readlines()
            if ignore_timestamp:
                lines1 = _remove_timestamps_from_lines(lines1)
                lines2 = _remove_timestamps_from_lines(lines2)
            for (line1, line2) in zip_longest(lines1, lines2, fillvalue=None):
                if (line1 != line2) or (None in (line1, line2)):
                    return (line1, line2)  #  type: ignore
    return None


def _remove_timestamps_from_lines(lines):
    # Assume the timestamp is contained in [], followed by a space and
    # at the beginning of the string. If so, it will be dropped
    # completely.
    return ["] ".join(line.split("] ")[1:]) for line in lines]


if __name__ == "__main__":
    # df = pd.read_csv("baseline_gridspace_feature-grid_50rep_srand.csv")
    # repeat_crossvalidation_baseline(
    #     setting=df.iloc[19],
    #     additional_settings=dict(param_privacy_budget=1.0),
    #     n_repetitions=100,
    #     cv=RepeatedKFold(n_splits=5, n_repeats=2),
    # )
    log_best_abalone_configurations()
    # dp_rmse_score_variation()
    # dp_rmse_score_variation_bun_steinke()
    # dp_rmse2_score_variation()
    # print(diff_logs("1.txt", "2.txt"))

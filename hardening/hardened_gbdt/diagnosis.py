from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import dpgbdt
from example_main import abalone_fit_arguments

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


def baseline(csv_filename: str, index: int):
    fit_args = abalone_fit_arguments()
    X = fit_args.pop("X")
    y = fit_args.pop("y")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df = pd.read_csv(csv_filename)
    params = params_from_series(df.loc[index])
    extended_params = extend_params(
        params,
        training_variant="vanilla",
        tree_scorer=None,
        verbosity="debug",
    )

    print(f"Parameters: {extended_params}")
    estimator = dpgbdt.DPGBDTRegressor(**extended_params)
    estimator.fit(X_train, y_train, **fit_args)
    print(f"fitted estimator: {estimator}")
    score = _rmse(estimator.predict(X_test), y_test)
    print(score)


def dp_rmse(csv_filename: str, index: int):
    fit_args = abalone_fit_arguments()
    X = fit_args.pop("X")
    y = fit_args.pop("y")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df = pd.read_csv(csv_filename)
    params = params_from_series(df.loc[index])
    extended_params = extend_params(
        params,
        training_variant="dp_argmax_scoring",
        tree_scorer="dp_rmse",
        verbosity="debug",
    )

    print(f"Parameters: {extended_params}")
    estimator = dpgbdt.DPGBDTRegressor(**extended_params)
    estimator.fit(X_train, y_train, **fit_args)
    print(f"fitted estimator: {estimator}")
    score = _rmse(estimator.predict(X_test), y_test)
    print(score)


def dp_quantile():
    return None


if __name__ == "__main__":
    # baseline("baseline_dense-gridspace_20221107_feature-grid.csv", 112872)
    # baseline("baseline_gridspace_20221109_feature-grid.csv", 3865)
    dp_rmse("dp_rmse_ts_gridspace_feature-grid.csv", 19221)

    # X, y, cat_idx, num_idx = get_abalone()
    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # # A simple baseline: mean of the training set
    # y_pred = np.mean(y_train).repeat(len(y_test))
    # print("Mean - RMSE: {0:f}".format(np.sqrt(np.mean(np.square(y_pred - y_test)))))

    # df = pd.read_csv("dp_rmse_ts_grid_eps100.0.csv")
    # line_of_interest = df[
    #     (df["rank_test_score"] == 1) & (df["param_privacy_budget"] == 100)
    # ].iloc[0]
    # print(line_of_interest)

    # params = params_from_series(line_of_interest)
    # params["dp_argmax_privacy_budget"] = 50.0
    # print(params)
    # estimator = dpgbdt.DPGBDTRegressor(**params)
    # estimator.fit(X_train, y_train, cat_idx, num_idx)
    # y_pred = estimator.predict(X_test)
    # print(
    #     "Depth first growth - RMSE: {0:f}".format(
    #         np.sqrt(np.mean(np.square(y_pred - y_test)))
    #     )
    # )

# -*- coding: utf-8 -*-
# gtheo@ethz.ch
# modified by: Moritz Kirschte, Yannik Potdevin
"""Example test file."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import dpgbdt
from sklearn.model_selection import train_test_split


def abalone_fit_arguments(n_rows: Optional[int] = None) -> dict[str, Any]:
    """Parse the abalone dataset and return parameters suitable for
    `fit`.

    Args:
      n_rows (int): Numbers of rows to read.

    Returns:
      dict[str, Any]: An assignment to `fit`'s arguments.
    """
    args = {}
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
    if n_rows:
        data = data.head(n_rows)
    args["y"] = data.rings.values.astype(float)
    del data["rings"]
    args["X"] = data.values.astype(float)
    args["cat_idx"] = [0]  # Sex
    args["num_idx"] = list(range(1, args["X"].shape[1]))  # Other attributes
    args["grid_lower_bounds"] = np.array(args["X"].shape[1] * [0.0])
    args["grid_upper_bounds"] = np.array([2.0, 1.0, 1.0, 1.5, 3.0, 2.0, 1.0, 1.5])
    args["grid_step_sizes"] = np.array([1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    return args


if __name__ == "__main__":
    fit_args = abalone_fit_arguments()
    X = fit_args.pop("X")
    y = fit_args.pop("y")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # A simple baseline: mean of the training set
    y_pred = np.mean(y_train).repeat(len(y_test))
    print("Mean - RMSE: {0:f}".format(np.sqrt(np.mean(np.square(y_pred - y_test)))))

    rng = dpgbdt.make_rng(42)
    tr = dpgbdt.make_tree_rejector(
        which="dp_rmse",
        n_trees_to_accept=1,
        U=100.0,
        gamma=2.0,
        rng=rng,
    )

    ts = dpgbdt.make_tree_scorer(
        which="dp_rmse",
        upper_bound=100.0,
        gamma=2.0,
        rng=rng,
    )

    # Train the model using a depth-first approach
    estimator = dpgbdt.DPGBDTRegressor(
        tree_rejector=tr,
        tree_scorer=ts,
    )
    estimator.fit(X_train, y_train, **fit_args)
    y_pred = estimator.predict(X_test)
    print(
        "Depth first growth - RMSE: {0:f}".format(
            np.sqrt(np.mean(np.square(y_pred - y_test)))
        )
    )

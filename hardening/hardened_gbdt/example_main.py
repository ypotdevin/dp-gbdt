# -*- coding: utf-8 -*-
# gtheo@ethz.ch
# modified by: Moritz Kirschte, Yannik Potdevin
"""Example test file."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import dpgbdt
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator


def get_abalone(n_rows: Optional[int] = None) -> Any:
    """Parse the abalone dataset.

    Args:
      n_rows (int): Numbers of rows to read.

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
    if n_rows:
        data = data.head(n_rows)
    y = data.rings.values.astype(float)
    del data["rings"]
    X = data.values.astype(float)
    cat_idx = [0]  # Sex
    num_idx = list(range(1, X.shape[1]))  # Other attributes
    return X, y, cat_idx, num_idx


if __name__ == "__main__":
    X, y, cat_idx, num_idx = get_abalone()
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
    #check_estimator(estimator)
    estimator.fit(X_train, y_train, cat_idx, num_idx)
    y_pred = estimator.predict(X_test)
    print(
        "Depth first growth - RMSE: {0:f}".format(
            np.sqrt(np.mean(np.square(y_pred - y_test)))
        )
    )

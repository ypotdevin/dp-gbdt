# -*- coding: utf-8 -*-
# gtheo@ethz.ch
# modified by: Moritz Kirschte, Yannik Potdevin
"""Example test file."""


import numpy as np
from sklearn.model_selection import train_test_split

import dpgbdt
from datasets.real.data_reader import abalone_fit_arguments

if __name__ == "__main__":
    fit_args = abalone_fit_arguments()
    X = fit_args.pop("X")
    y = fit_args.pop("y")
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # A simple baseline: mean of the training set
    y_pred = np.mean(y_train).repeat(len(y_test))
    print("Mean - RMSE: {0:f}".format(np.sqrt(np.mean(np.square(y_pred - y_test)))))

    rng = dpgbdt.make_rng(42)
    ts = dpgbdt.make_tree_scorer(
        which="dp_rmse",
        upper_bound=100.0,
        gamma=2.0,
        rng=rng,
    )

    # Train the model using a depth-first approach
    estimator = dpgbdt.DPGBDTRegressor(
        tree_scorer=ts,
    )
    estimator.fit(X_train, y_train, **fit_args)
    y_pred = estimator.predict(X_test)
    print(
        "Depth first growth - RMSE: {0:f}".format(
            np.sqrt(np.mean(np.square(y_pred - y_test)))
        )
    )

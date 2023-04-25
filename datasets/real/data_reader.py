from typing import Any, Optional

import numpy as np
import pandas as pd


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
    args["grid_lower_bounds"] = np.zeros(args["X"].shape[1])
    args["grid_upper_bounds"] = np.array([2.0, 1.0, 1.0, 1.5, 3.0, 2.0, 1.0, 1.5])
    args["grid_step_sizes"] = np.array([1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    return args


def wine_fit_arguments(n_rows: Optional[int] = None) -> dict[str, Any]:
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
    if n_rows:
        data = data.head(n_rows)
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


def concrete_fit_arguments(n_rows: Optional[int] = None) -> dict[str, Any]:
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
    if n_rows:
        data = data.head(n_rows)
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


def metro_fit_arguments(n_rows: Optional[int] = None) -> dict[str, Any]:
    args = {}
    data = pd.read_csv(
        "./datasets/real/Metro_Interstate_Traffic_Volume_light.csv",
        index_col=0,
    )
    if n_rows is not None:
        data = data.head(n_rows)
    args["y"] = data.traffic_volume.values.astype(float)
    del data["traffic_volume"]
    args["X"] = data.values.astype(float)
    args["cat_idx"] = []
    args["num_idx"] = list(range(args["X"].shape[1]))
    # the ordering of the columns is ["temp", "year", "day", "weekday", "hour"]
    args["grid_lower_bounds"] = np.array([200.0, 2012.0, 1.0, 0.0, 0.0])
    args["grid_upper_bounds"] = np.array([350.0, 2018.0, 31.0, 6.0, 23.0])
    args["grid_step_sizes"] = np.array([0.1, 1.0, 1.0, 1.0, 1.0])
    return args

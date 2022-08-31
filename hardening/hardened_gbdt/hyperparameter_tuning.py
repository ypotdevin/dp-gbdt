import socket
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from ray.tune.sklearn import TuneSearchCV

import dpgbdt


def tune(
    regressor,
    data_provider: Callable[[], Tuple[np.ndarray, np.ndarray, list[int], list[int]]],
    parameter_grid: dict[str, Any],
    n_trials: int,
    n_jobs: Optional[int] = None,
    time_budget_s: Optional[int] = None,
    search_optimization="bayesian",
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    X_train, y_train, cat_idx, num_idx = data_provider()
    tune_search = TuneSearchCV(
        regressor,
        parameter_grid,
        search_optimization=search_optimization,
        n_trials=n_trials,
        n_jobs=n_jobs,
        time_budget_s=time_budget_s,
        scoring="neg_root_mean_squared_error",
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


if __name__ == "__main__":
    tr_rng = dpgbdt.make_rng()
    rejector_params = dict(rejection_budget=0.05, gamma=2.0, rng=tr_rng)
    tree_rejectors = [
        dpgbdt.make_tree_rejector("dp_rmse", **{"U": U, **rejector_params})
        for U in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    ]
    regressor = dpgbdt.DPGBDTRegressor()
    parameter_grid = dict(
        tree_rejector=tree_rejectors,
        learning_rate=(0.0, 10.0),
        nb_trees=(1, 50),
        max_depth=(1, 10),
        l2_threshold=(0.01, 10.0),
        l2_lambda=(0.1, 10.0),
    )
    df, best_params = tune(
        regressor,
        get_abalone,
        parameter_grid,
        n_trials=1000,
        time_budget_s=60 * 60 * 36,  # 36 hours
    )
    df.to_csv(f"{socket.gethostname()}.csv")
    print(f"Best parameters: {best_params}")

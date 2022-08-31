from typing import Optional
This is a module to be used as a reference for building other modules
"""
from numpy.random import default_rng
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

import pyestimator


class DPGBDTRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        seed: Optional[int] = None,
        privacy_budget: float = 1.0,
        tree_rejector=None,
        learning_rate: float = 5.0,
        nb_trees: int = 1,
        max_depth: int = 5,
        min_samples_split: int = 2,
        l2_threshold: float = 0.3,
        l2_lambda: float = 7.7,
        balance_partition: bool = True,
        gradient_filtering: bool = True,
        leaf_clipping: bool = True,
        use_decay: bool = False,
    ):
        self.rng = rng
        self.privacy_budget = privacy_budget
        self.tree_rejector = tree_rejector
        self.learning_rate = learning_rate
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2_threshold = l2_threshold
        self.l2_lambda = l2_lambda
        self.balance_partition = balance_partition
        self.gradient_filtering = gradient_filtering
        self.leaf_clipping = leaf_clipping
        self.use_decay = use_decay

    def fit(self, X, y, cat_idx=None, num_idx=None):
        """_description_

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        cat_idx : list of int
            The indices of categorical (non-numerical) value columns.
            Defaults to empty list.
        num_idx : list of int
            The indices of numerical value columns. Defaults to all
            columns.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, order="C")
        X, y = _ensure_float(X, y)

        self.rng_ = make_rng(self.seed)
        if self.tree_rejector is None:
            self.tree_rejector = make_tree_rejector("constant", decision=False)
        if cat_idx is None:
            cat_idx = []
        if num_idx is None:
            num_idx = list(range(len(X[0])))

        self.estimator_ = pyestimator.PyEstimator(
            rng=self.rng_,
            privacy_budget=self.privacy_budget,
            tree_rejector=self.tree_rejector,
            learning_rate=self.learning_rate,
            nb_trees=self.nb_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            l2_threshold=self.l2_threshold,
            l2_lambda=self.l2_lambda,
            balance_partition=self.balance_partition,
            gradient_filtering=self.gradient_filtering,
            leaf_clipping=self.leaf_clipping,
            use_decay=self.use_decay,
        )
        self.estimator_.fit(X, y, cat_idx, num_idx)

        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            _description_
        """
        check_is_fitted(self, ["estimator_", "rng_"])
        X = check_array(X, order="C")
        X = _ensure_float(X)

        return self.estimator_.predict(X)

    def score(self, X, y, sample_weight=None):
        """_summary_

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        r_squared : float
            The R^2 score (coefficient of determination).
        """
        check_is_fitted(self, ["estimator_", "rng_"])
        X, y = check_X_y(X, y, order="C")
        X, y = _ensure_float(X, y)

        return r2_score(y, self.predict(X), sample_weight=sample_weight)


def _ensure_float(X, y=None):
    if not X.dtype == float:
        try:
            print(
                f"Array X is of type {X.dtype} instead of float64. Casting to float64."
            )
            X = X.astype("float64", casting="safe")
        except:
            raise ValueError("Casting X safely to float array failed.")
    if y is not None and not y.dtype == float:
        try:
            print(
                f"Array y is of type {y.dtype} instead of float64. Casting to float64."
            )
            y = y.astype("float64", casting="safe")
        except:
            raise ValueError("Casting X safely to float array failed.")

    if y is not None:
        return (X, y)
    else:
        return X


def make_rng(seed: Optional[int] = None) -> pyestimator.PyMT19937:
    if seed is None:
        rng = default_rng()
        # Stay within the the bounds of mt19937's input seed (C++)
        seed = rng.integers(2 ** 30 - 1)
    return pyestimator.PyMT19937(seed)


def make_tree_rejector(which: str, **kwargs):
    selector = dict(
        constant=pyestimator.PyConstantRejector,
        quantile_linear_combination=pyestimator.PyQuantileLinearCombinationRejector,
        dp_rmse=pyestimator.PyDPrMSERejector,
        approximate_dp_rmse=pyestimator.PyApproxDPrMSERejector,
    )
    return selector[which](**kwargs)

# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cimport cython
cimport numpy as np
from cpp_estimator cimport (Estimator, TreeScorer, DPrMSEScorer, DPQuantileScorer,
                            TreeRejector, ConstantRejector,
                            QuantileLinearCombinationRejector,
                            DPrMSERejector, ApproxDPrMSERejector, mt19937)
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np


cdef class PyMT19937:
    cdef mt19937 c_rng
    cdef int seed

    def __cinit__(self, int seed):
        self.c_rng = mt19937(seed)
        self.seed = seed

    # see https://stackoverflow.com/a/18611983 (to enable deepcopy for sklearn)
    def __reduce__(self):
        return (self.__class__, (self.seed,))

    def __repr__(self):
        return f"PyMT19937(seed={self.seed})"

cdef class PyTreeScorer:
    cdef shared_ptr[TreeScorer] sptr_ts

cdef class PyDPrMSEScorer(PyTreeScorer):
    cdef double upper_bound, gamma
    cdef PyMT19937 rng

    def __cinit__(self, double upper_bound, double gamma, PyMT19937 rng):
        self.upper_bound = upper_bound
        self.gamma = gamma
        self.rng = rng
        self.sptr_ts = shared_ptr[TreeScorer](new DPrMSEScorer(upper_bound, gamma, rng.c_rng))

    def __reduce__(self):
        return (self.__class__, (self.upper_bound, self.gamma, self.rng))

    def __repr__(self):
        return f"PyDPrMSEScorer(upper_bound={self.upper_bound},"\
               f"gamma={self.gamma},rng={self.rng})"

cdef class PyDPQuantileScorer(PyTreeScorer):
    cdef double shift, scale, upper_bound
    cdef list qs
    cdef PyMT19937 rng

    def __cinit__(
        self,
        double shift,
        double scale,
        list qs,
        double upper_bound,
        PyMT19937 rng
    ):
        self.shift = shift
        self.scale = scale
        self.qs = qs
        self.upper_bound = upper_bound
        self.rng = rng
        cdef vector[double] qs_vec = qs
        self.sptr_ts = shared_ptr[TreeScorer](
            new DPQuantileScorer(shift, scale, qs_vec, upper_bound, rng.c_rng)
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.shift, self.scale, self.qs, self.upper_bound, self.rng)
        )

    def __repr__(self):
        return f"PyDPQuantileScorer(shift={self.shift},"\
               f"scale={self.scale},"\
               f"qs={self.qs},"\
               f"upper_bound={self.upper_bound},"\
               f"rng={self.rng})"

cdef class PyTreeRejector:
    cdef shared_ptr[TreeRejector] sptr_tr


cdef class PyConstantRejector(PyTreeRejector):
    cdef bool decision

    def __cinit__(self, bool decision):
        self.decision = decision
        self.sptr_tr = shared_ptr[TreeRejector](new ConstantRejector(decision))

    def __reduce__(self):
        return (self.__class__, (self.decision,))

    def __repr__(self):
        return f"PyConstantRejector(decision={self.decision})"

cdef class PyQuantileLinearCombinationRejector(PyTreeRejector):
    cdef list qs, coefficients

    def __cinit__(self, list qs, list coefficients):
        self.qs = qs
        self.coefficients = coefficients
        cdef vector[double] qs_vec = qs
        cdef vector[double] coefficients_vec = coefficients
        self.sptr_tr = shared_ptr[TreeRejector](
            new QuantileLinearCombinationRejector(qs_vec, coefficients_vec)
        )

    def __reduce__(self):
        return (self.__class__, (self.qs, self.coefficients))

    def __repr__(self):
        return f"PyQuantileLinearCombinationRejector(qs={self.qs},"\
               f"coefficients={self.coefficients})"

cdef class PyDPrMSERejector(PyTreeRejector):
    cdef int n_trees_to_accept
    cdef double U, gamma
    cdef PyMT19937 rng

    def __cinit__(self, int n_trees_to_accept, double U, double gamma, PyMT19937 rng):
        self.n_trees_to_accept = n_trees_to_accept
        self.U = U
        self.gamma = gamma
        self.rng = rng
        self.sptr_tr = shared_ptr[TreeRejector](
            new DPrMSERejector(n_trees_to_accept, U, gamma, rng.c_rng)
        )

    def __reduce__(self):
        return (self.__class__, (self.n_trees_to_accept, self.U, self.gamma, self.rng))

    def __repr__(self):
        return f"PyDPrMSERejector(n_trees_to_accept={self.n_trees_to_accept},"\
               f"U={self.U},"\
               f"gamma={self.gamma},"\
               f"rng={self.rng})"

cdef class PyApproxDPrMSERejector(PyTreeRejector):
    cdef int n_trees_to_accept
    cdef double delta, U
    cdef PyMT19937 rng

    def __cinit__(self, int n_trees_to_accept, double delta, double U, PyMT19937 rng):
        self.n_trees_to_accept = n_trees_to_accept
        self.delta = delta
        self.U = U
        self.rng = rng
        self.sptr_tr = shared_ptr[TreeRejector](
            new ApproxDPrMSERejector(n_trees_to_accept, delta, U, rng.c_rng)
        )

    def __reduce__(self):
        return (self.__class__, (self.n_trees_to_accept, self.delta, self.U, self.rng))

    def __repr__(self):
        return f"PyApproxDPrMSERejector(n_trees_to_accept={self.n_trees_to_accept},"\
               f"delta={self.delta},"\
               f"U={self.U},"\
               f"gamma={self.gamma},"\
               f"rng={self.rng})"

cdef class PyEstimator:
    cdef Estimator* estimator
    cdef PyMT19937 rng
    cdef double privacy_budget, ensemble_rejector_budget_split, dp_argmax_privacy_budget, dp_argmax_stopping_prob, learning_rate, l2_threshold, l2_lambda
    cdef string training_variant, verbosity
    cdef PyTreeRejector tree_rejector
    cdef PyTreeScorer tree_scorer
    cdef int n_trees_to_accept, max_depth, min_samples_split
    cdef bool balance_partition, gradient_filtering, leaf_clipping, use_decay

    def __cinit__(
        self,
        PyMT19937 rng,
        double privacy_budget,
        double ensemble_rejector_budget_split,
        str training_variant,
        PyTreeRejector tree_rejector,
        double dp_argmax_privacy_budget,
        double dp_argmax_stopping_prob,
        PyTreeScorer tree_scorer,
        double learning_rate,
        int n_trees_to_accept,
        int max_depth,
        int min_samples_split,
        double l2_threshold,
        double l2_lambda,
        bool balance_partition,
        bool gradient_filtering,
        bool leaf_clipping,
        bool use_decay,
        str verbosity
    ):
        self.rng = rng
        self.privacy_budget = privacy_budget
        self.ensemble_rejector_budget_split = ensemble_rejector_budget_split
        self.training_variant = training_variant.encode("UTF-8") # converting to bytes
        self.tree_rejector = tree_rejector
        self.tree_scorer = tree_scorer
        self.dp_argmax_privacy_budget = dp_argmax_privacy_budget
        self.dp_argmax_stopping_prob = dp_argmax_stopping_prob
        self.learning_rate = learning_rate
        self.n_trees_to_accept = n_trees_to_accept
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.l2_threshold = l2_threshold
        self.l2_lambda = l2_lambda
        self.balance_partition = balance_partition
        self.gradient_filtering = gradient_filtering
        self.leaf_clipping = leaf_clipping
        self.use_decay = use_decay
        self.verbosity = verbosity.encode("UTF-8")
        self.estimator = new Estimator(
            rng=rng.c_rng,
            privacy_budget=privacy_budget,
            ensemble_rejector_budget_split=ensemble_rejector_budget_split,
            training_variant=self.training_variant,
            tree_rejector=tree_rejector.sptr_tr,
            tree_scorer=tree_scorer.sptr_ts,
            dp_argmax_privacy_budget=dp_argmax_privacy_budget,
            dp_argmax_stopping_prob=dp_argmax_stopping_prob,
            learning_rate=learning_rate,
            n_trees_to_accept=n_trees_to_accept,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            l2_threshold=l2_threshold,
            l2_lambda=l2_lambda,
            balance_partition=balance_partition,
            gradient_filtering=gradient_filtering,
            leaf_clipping=leaf_clipping,
            use_decay=use_decay,
            verbosity=self.verbosity,
        )


    def __dealloc__(self):
        del self.estimator

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.rng,
                self.privacy_budget,
                self.ensemble_rejector_budget_split,
                self.training_variant,
                self.tree_rejector,
                self.dp_argmax_privacy_budget,
                self.dp_argmax_stopping_prob,
                self.tree_scorer,
                self.learning_rate,
                self.n_trees_to_accept,
                self.max_depth,
                self.min_samples_split,
                self.l2_threshold,
                self.l2_lambda,
                self.balance_partition,
                self.gradient_filtering,
                self.leaf_clipping,
                self.use_decay,
                self.verbosity,
            )
        )

    def __repr__(self):
        return f"PyEstimator(rng={self.rejection_budget},"\
               f"privacy_budget={self.privacy_budget},"\
               f"ensemble_rejector_budget_split={self.ensemble_rejector_budget_split},"\
               f"training_variant={self.training_variant},"\
               f"tree_rejector={self.tree_rejector},"\
               f"dp_argmax_privacy_budget={self.dp_argmax_privacy_budget},"\
               f"dp_argmax_stopping_prob={self.dp_argmax_stopping_prob},"\
               f"tree_scorer={self.tree_scorer},"\
               f"learning_rate={self.learning_rate},"\
               f"n_trees_to_accept={self.n_trees_to_accept},"\
               f"max_depth={self.max_depth},"\
               f"min_samples_split={self.min_samples_split},"\
               f"l2_threshold={self.l2_threshold},"\
               f"l2_lambda={self.l2_lambda},"\
               f"balance_partition={self.balance_partition},"\
               f"gradient_filtering={self.gradient_filtering},"\
               f"leaf_clipping={self.leaf_clipping},"\
               f"use_decay={self.use_decay},"\
               f"verbosity={self.verbosity})"

    def fit(
        self,
        np.ndarray[double, ndim=2, mode="c"] X not None,
        np.ndarray[double, ndim=1, mode="c"] y not None,
        list cat_idx,
        list num_idx,
    ):
        cdef vector[vector[double]] x_vec = X
        cdef vector[double] y_vec = y
        cdef vector[int] cat_idx_vec = cat_idx
        cdef vector[int] num_idx_vec = num_idx
        self.estimator.fit(X, y, cat_idx_vec, num_idx_vec)
        return self

    def predict(self, np.ndarray[double, ndim=2, mode="c"] X not None):
        cdef vector[vector[double]] x_vec = X
        cdef vector[double] y_pred
        y_pred = self.estimator.predict(x_vec)
        return np.asarray(y_pred)

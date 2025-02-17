# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cimport cython
cimport numpy as np
from cpp_estimator cimport (Estimator,
                            beta_smooth_sensitivity,
                            TreeScorer, DPrMSEScorer, DPrMSEScorer2,
                            DPQuantileScorer, BunSteinkeScorer,
                            PrivacyBucketScorer, LeakyRmseScorer,
                            mt19937,
                            Beta, ConstantBeta)
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

cdef class PyBeta:
    cdef shared_ptr[Beta] sptr_b

    def beta(self, double privacy_budget, double relaxation):
        return self.sptr_b.get().beta(privacy_budget, relaxation)

cdef class PyConstantBeta(PyBeta):
    cdef double beta_constant

    def __cinit__(self, double beta):
        self.beta_constant = beta
        self.sptr_b = shared_ptr[Beta](new ConstantBeta(beta))

    def __reduce__(self):
        return (self.__class__, (self.beta_constant))

    def __repr__(self):
        return f"PyConstantBeta(beta={self.beta_constant})"

cdef class PyTreeScorer:
    cdef shared_ptr[TreeScorer] sptr_ts

    def score_tree(
        self,
        double privacy_budget,
        np.ndarray[double, ndim=1, mode="c"] y not None,
        np.ndarray[double, ndim=1, mode="c"] y_pred not None,
    ):
        cdef vector[double] y_vec = y
        cdef vector[double] y_pred_vec = y_pred
        return self.sptr_ts.get().score_tree(privacy_budget, y_vec, y_pred_vec)

cdef class PyLeakyRmseScorer(PyTreeScorer):

    def __cinit__(self):
        self.sptr_ts = shared_ptr[TreeScorer](new LeakyRmseScorer())

    def __reduce__(self):
        return (self.__class__,)

    def __repr__(self):
        return f"PyLeakyRmseScorer()"

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

cdef class PyDPrMSEScorer2(PyTreeScorer):
    cdef PyBeta beta
    cdef double upper_bound, gamma
    cdef PyMT19937 rng

    def __cinit__(
        self,
        PyBeta beta,
        double upper_bound,
        double gamma,
        PyMT19937 rng
    ):
        self.beta = beta
        self.upper_bound = upper_bound
        self.gamma = gamma
        self.rng = rng
        self.sptr_ts = shared_ptr[TreeScorer](
            new DPrMSEScorer2(
                beta.sptr_b,
                upper_bound,
                gamma,
                rng.c_rng
            )
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.beta, self.upper_bound, self.gamma, self.rng)
        )

    def __repr__(self):
        return f"PyDPrMSEScorer2(beta={self.beta},"\
               f"upper_bound={self.upper_bound},"\
               f"gamma={self.gamma},"\
               f"rng={self.rng})"

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


cdef class PyBunSteinkeScorer(PyTreeScorer):
    cdef double upper_bound, beta, relaxation
    cdef PyMT19937 rng

    def __cinit__(
        self,
        double upper_bound,
        double beta,
        double relaxation,
        PyMT19937 rng
    ):
        self.upper_bound = upper_bound
        self.beta = beta
        self.relaxation = relaxation
        self.rng = rng
        self.sptr_ts = shared_ptr[TreeScorer](
            new BunSteinkeScorer(upper_bound, beta, relaxation, rng.c_rng)
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.upper_bound, self.beta, self.relaxation, self.rng)
        )

    def __repr__(self):
        return f"PyBunSteinkeScorer(upper_bound={self.upper_bound},"\
               f"beta={self.beta},"\
               f"relaxation={self.relaxation},"\
               f"rng={self.rng})"

cdef class PyPrivacyBucketScorer(PyTreeScorer):
    cdef double upper_bound, beta
    cdef int n_trees_to_accept
    cdef list coefficients
    cdef PyMT19937 rng

    def __cinit__(
        self,
        double upper_bound,
        double beta,
        int n_trees_to_accept,
        list coefficients,
        PyMT19937 rng
    ):
        self.upper_bound = upper_bound
        self.beta = beta
        self.n_trees_to_accept = n_trees_to_accept
        self.coefficients = coefficients
        cdef vector[double] coefficients_vec = coefficients
        self.rng = rng
        self.sptr_ts = shared_ptr[TreeScorer](
            new PrivacyBucketScorer(
                upper_bound,
                beta,
                n_trees_to_accept,
                coefficients_vec,
                rng.c_rng
            )
        )

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.upper_bound,
                self.beta,
                self.n_trees_to_accept,
                self.coefficients,
                self.rng
            )
        )

    def __repr__(self):
        return f"PyPrivacyBucketScorer(uppen_bound={self.upper_bound},"\
               f"beta={self.beta},"\
               f"n_trees_to_accept={self.n_trees_to_accept},"\
               f"coefficients={self.coefficients},"\
               f"rng={self.rng})"

cdef class PyEstimator:
    cdef Estimator* estimator
    cdef PyMT19937 rng
    cdef double privacy_budget, ensemble_rejector_budget_split, dp_argmax_privacy_budget, dp_argmax_stopping_prob, learning_rate, l2_threshold, l2_lambda
    cdef string training_variant, verbosity
    cdef PyTreeScorer tree_scorer
    cdef int n_trees_to_accept, max_depth, min_samples_split
    cdef bool balance_partition, gradient_filtering, leaf_clipping, use_decay

    def __cinit__(
        self,
        PyMT19937 rng,
        double privacy_budget,
        double ensemble_rejector_budget_split,
        str training_variant,
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
                self.tree_scorer,
                self.dp_argmax_privacy_budget,
                self.dp_argmax_stopping_prob,
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
               f"tree_scorer={self.tree_scorer},"\
               f"dp_argmax_privacy_budget={self.dp_argmax_privacy_budget},"\
               f"dp_argmax_stopping_prob={self.dp_argmax_stopping_prob},"\
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
        np.ndarray[double, ndim=1, mode="c"] grid_lower_bounds not None,
        np.ndarray[double, ndim=1, mode="c"] grid_upper_bounds not None,
        np.ndarray[double, ndim=1, mode="c"] grid_step_sizes not None,
    ):
        cdef vector[vector[double]] x_vec = X
        cdef vector[double] y_vec = y
        cdef vector[int] cat_idx_vec = cat_idx
        cdef vector[int] num_idx_vec = num_idx
        cdef vector[double] grid_lower_bounds_vec = grid_lower_bounds
        cdef vector[double] grid_upper_bounds_vec = grid_upper_bounds
        cdef vector[double] grid_step_sizes_vec = grid_step_sizes
        self.estimator.fit(
            X,
            y,
            cat_idx_vec,
            num_idx_vec,
            grid_lower_bounds_vec,
            grid_upper_bounds_vec,
            grid_step_sizes_vec
        )
        return self

    def predict(self, np.ndarray[double, ndim=2, mode="c"] X not None):
        cdef vector[vector[double]] x_vec = X
        cdef vector[double] y_pred
        y_pred = self.estimator.predict(x_vec)
        return np.asarray(y_pred)

def py_beta_smooth_sensitivity(
    np.ndarray[double, ndim=1, mode="c"] errors not None,
    beta: cython.double,
    upper_bound: cython.double
) -> cython.double:
    cdef vector[double] errors_vec = errors
    return beta_smooth_sensitivity(errors_vec, beta, upper_bound)
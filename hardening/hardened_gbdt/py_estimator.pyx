# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cimport cython
cimport numpy as np
from cpp_estimator cimport (Estimator, TreeRejector, ConstantRejector,
                            QuantileLinearCombinationRejector, DPrMSERejector,
                            ApproxDPrMSERejector, mt19937)
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

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

cdef class PyTreeRejector:
    cdef shared_ptr[TreeRejector] sptr_tr


cdef class PyConstantRejector(PyTreeRejector):
    cdef bool decision

    def __cinit__(self, bool decision):
        self.decision = decision
        self.sptr_tr = shared_ptr[TreeRejector](new ConstantRejector(decision))

    def __reduce__(self):
        return (self.__class__, (self.decision,))

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


cdef class PyDPrMSERejector(PyTreeRejector):
    cdef double rejection_budget, U, gamma
    cdef PyMT19937 rng

    def __cinit__(self, double rejection_budget, double U, double gamma, PyMT19937 rng):
        self.rejection_budget = rejection_budget
        self.U = U
        self.gamma = gamma
        self.rng = rng
        self.sptr_tr = shared_ptr[TreeRejector](
            new DPrMSERejector(rejection_budget, U, gamma, rng.c_rng)
        )

    def __reduce__(self):
        return (self.__class__, (self.rejection_budget, self.U, self.gamma, self.rng))

cdef class PyApproxDPrMSERejector(PyTreeRejector):
    cdef double rejection_budget, delta, U
    cdef PyMT19937 rng

    def __cinit__(self, double rejection_budget, double delta, double U, PyMT19937 rng):
        self.rejection_budget = rejection_budget
        self.delta = delta
        self.U = U
        self.rng = rng
        self.sptr_tr = shared_ptr[TreeRejector](
            new ApproxDPrMSERejector(rejection_budget, delta, U, rng.c_rng)
        )

    def __reduce__(self):
        return (self.__class__, (self.rejection_budget, self.delta, self.U, self.rng))


cdef class PyEstimator:
    cdef Estimator* estimator
    cdef PyMT19937 rng
    cdef double privacy_budget, learning_rate, l2_threshold, l2_lambda
    cdef PyTreeRejector tree_rejector
    cdef int nb_trees, max_depth, min_samples_split
    cdef bool balance_partition, gradient_filtering, leaf_clipping, use_decay

    def __cinit__(
        self,
        PyMT19937 rng,
        double privacy_budget,
        PyTreeRejector tree_rejector,
        double learning_rate,
        int nb_trees,
        int max_depth,
        int min_samples_split,
        double l2_threshold,
        double l2_lambda,
        bool balance_partition,
        bool gradient_filtering,
        bool leaf_clipping,
        bool use_decay
    ):
        self.rng = rng
        self.privacy_budget = privacy_budget
        self.tree_rejector = tree_rejector
        self.learning_rate = learning_rate
        self.nb_trees = nb_trees
        self.max_depth = max_depth
        self. min_samples_split = min_samples_split
        self.l2_threshold = l2_threshold
        self.l2_lambda = l2_lambda
        self.balance_partition = balance_partition
        self.gradient_filtering = gradient_filtering
        self.leaf_clipping = leaf_clipping
        self.use_decay = use_decay
        self.estimator = new Estimator(
            rng=rng.c_rng,
            privacy_budget=privacy_budget,
            tree_rejector=tree_rejector.sptr_tr,
            learning_rate=learning_rate,
            nb_trees=nb_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            l2_threshold=l2_threshold,
            l2_lambda=l2_lambda,
            balance_partition=balance_partition,
            gradient_filtering=gradient_filtering,
            leaf_clipping=leaf_clipping,
            use_decay=use_decay,
        )


    def __dealloc__(self):
        del self.estimator

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.rng,
                self.privacy_budget,
                self.tree_rejector,
                self.learning_rate,
                self.nb_trees,
                self.max_depth,
                self.min_samples_split,
                self.l2_threshold,
                self.l2_lambda,
                self.balance_partition,
                self.gradient_filtering,
                self.leaf_clipping,
                self.use_decay,
            )
        )

    def fit(
        self,
        np.ndarray[double, ndim=2, mode="c"] X not None,
        np.ndarray[double, ndim=1, mode="c"] y not None,
        list cat_idx,
        list num_idx
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

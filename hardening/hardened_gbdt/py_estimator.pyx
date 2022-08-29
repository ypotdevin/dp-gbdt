# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

cimport cython
cimport numpy as np
from cpp_estimator cimport (Estimator, TreeRejector, DPrMSERejector,
                            QuantileLinearCombinationRejector, mt19937)
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector

import numpy as np


cdef class PyMT19937:
    cdef mt19937 c_rng

    def __cinit__(self, int seed):
        self.c_rng = mt19937(seed)


cdef class PyTreeRejector:
    cdef shared_ptr[TreeRejector] sptr_tr


cdef class PyDPrMSERejector(PyTreeRejector):
    def __cinit__(self, double epsilon, double U, double gamma, PyMT19937 rng):
        self.sptr_tr = shared_ptr[TreeRejector](new DPrMSERejector(epsilon, U, gamma, rng.c_rng))


cdef class PyQuantileLinearCombinationRejector(PyTreeRejector):
    def __cinit__(self, list qs, list coefficients):
        cdef vector[double] qs_vec = qs
        cdef vector[double] coefficients_vec = coefficients
        self.sptr_tr = shared_ptr[TreeRejector](new QuantileLinearCombinationRejector(qs, coefficients))


cdef class PyEstimator:
    cdef Estimator* estimator

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

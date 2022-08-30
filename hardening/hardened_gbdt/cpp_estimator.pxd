"""DP-GBDT namespace declarations."""

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "estimator.h" namespace "dpgbdt":
    cdef cppclass Estimator:
        Estimator() except +
        Estimator(
            mt19937 rng,
            double privacy_budget,
            shared_ptr[TreeRejector] tree_rejector,
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
        ) except +
        Estimator fit(vector[vector[double]] X, vector[double] y, vector[int] cat_idx, vector[int] num_idx)
        vector[double] predict(vector[vector[double]] X)

cdef extern from "tree_rejection.h" namespace "tree_rejection":
    cdef cppclass TreeRejector:
        pass

    cdef cppclass ConstantRejector(TreeRejector):
        ConstantRejector(bool decision) except +

    cdef cppclass DPrMSERejector(TreeRejector):
        DPrMSERejector(
            double rejection_budget, double U, double gamma, mt19937 rng
        ) except +

    cdef cppclass QuantileLinearCombinationRejector(TreeRejector):
        QuantileLinearCombinationRejector(
            vector[double] qs, vector[double] coefficients
        ) except +

    cdef cppclass ApproxDPrMSERejector(TreeRejector):
        ApproxDPrMSERejector(
            double rejection_budget, double delta, double U, mt19937 rng
        ) except +

# see https://stackoverflow.com/a/40992452
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int seed) except +
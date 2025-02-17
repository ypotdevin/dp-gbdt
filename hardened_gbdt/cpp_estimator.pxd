"""DP-GBDT namespace declarations."""

from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool


cdef extern from "estimator.h" namespace "dpgbdt":
    cdef cppclass Estimator:
        Estimator() except +
        Estimator(
            mt19937 rng,
            double privacy_budget,
            double ensemble_rejector_budget_split,
            string training_variant,
            shared_ptr[TreeScorer] tree_scorer,
            double dp_argmax_privacy_budget,
            double dp_argmax_stopping_prob,
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
            string verbosity
        ) except +
        Estimator fit(
            vector[vector[double]] X,
            vector[double] y,
            vector[int] cat_idx,
            vector[int] num_idx,
            vector[double] grid_lower_bounds,
            vector[double] grid_upper_bounds,
            vector[double] grid_step_sizes
        ) except +
        vector[double] predict(vector[vector[double]] X)

cdef extern from "gbdt/loss.h":
    double beta_smooth_sensitivity(
        vector[double] errors,
        double beta,
        double U
    )

cdef extern from "gbdt/tree_rejection.h" namespace "tree_rejection":
    cdef cppclass Beta:
        double beta(double privacy_budget, double relaxation)

    cdef cppclass ConstantBeta(Beta):
        ConstantBeta(double beta) except +

    cdef cppclass TreeScorer:
        double score_tree(double privacy_budget, vector[double] y, vector[double] y_pred)

    cdef cppclass LeakyRmseScorer(TreeScorer):
        LeakyRmseScorer() except +

    cdef cppclass DPrMSEScorer(TreeScorer):
        DPrMSEScorer(double upper_bound, double gamma, mt19937 rng) except +

    cdef cppclass DPrMSEScorer2(TreeScorer):
        DPrMSEScorer2(
            shared_ptr[Beta] beta,
            double upper_bound,
            double gamma,
            mt19937 rng
        ) except +

    cdef cppclass DPQuantileScorer(TreeScorer):
        DPQuantileScorer(
            double shift,
            double scale,
            vector[double]qs,
            double upper_bound,
            mt19937 rng
        ) except +

    cdef cppclass BunSteinkeScorer(TreeScorer):
        BunSteinkeScorer(
            double upper_bound,
            double beta,
            double relaxation,
            mt19937 rng
        ) except +

    cdef cppclass PrivacyBucketScorer(TreeScorer):
        PrivacyBucketScorer(
            double upper_bound,
            double beta,
            int n_trees_to_accept,
            vector[double] coefficients,
            mt19937 &rng
        ) except +

# see https://stackoverflow.com/a/40992452
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() except +
        mt19937(unsigned int seed) except +
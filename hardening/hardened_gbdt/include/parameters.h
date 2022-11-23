#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory>
#include <random>
#include <vector>
#include "loss.h"
#include "utils.h"
#include "tree_rejection.h"

struct ModelParams
{
    // primary model parameters
    std::mt19937 rng;
    int n_trees_to_accept = 5;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    std::string training_variant = "dp_argmax_scoring";
    double ensemble_rejector_budget_split = 0.9;
    std::shared_ptr<tree_rejection::TreeRejector> tree_rejector;
    std::shared_ptr<tree_rejection::TreeScorer> tree_scorer;
    double dp_argmax_privacy_budget = 0.1;
    double stopping_prob = 0.05;
    int max_depth = 6;
    int min_samples_split = 2;
    unsigned balance_partition = true;
    unsigned gradient_filtering = true;
    unsigned leaf_clipping = true;
    unsigned use_decay = false;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;

    // secondary model parameters
    std::vector<double> grid_lower_bounds;
    std::vector<double> grid_upper_bounds;
    std::vector<double> grid_step_sizes;
    std::vector<std::vector<double>> cat_values; // deprecated
    unsigned scale_X = false;
    double scale_X_percentile = 95;
    double scale_X_privacy_budget = 0.4;
    unsigned scale_y = false;

    // dataset specific parameters
    std::shared_ptr<Task> task;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
};

std::ostream &operator<<(std::ostream &os, const ModelParams &mp);

// each tree has these additional parameters
struct TreeParams
{
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};

std::ostream &operator<<(std::ostream &os, const TreeParams &tp);

#endif /* PARAMETERS_H */

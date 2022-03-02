#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <memory>
#include <vector>
#include "utils.h"
#include "loss.h"

struct ModelParams
{
    int nb_trees;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    double optimization_privacy_budget = 1.0;
    std::shared_ptr<Task> task;
    int max_depth = 6;
    int min_samples_split = 2;
    unsigned balance_partition = HAMMING_TRUE;
    unsigned gradient_filtering = HAMMING_TRUE;
    unsigned leaf_clipping = HAMMING_TRUE;
    unsigned scale_y = HAMMING_FALSE;
    unsigned use_decay = HAMMING_FALSE;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;
    double error_upper_bound = 10.0;
    int verbosity = -1;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
    unsigned use_grid = HAMMING_FALSE;
    std::tuple<double, double> grid_borders;
    double grid_step_size;
    std::vector<std::vector<double>> cat_values;
    unsigned scale_X = HAMMING_FALSE;
    double scale_X_percentile = 95;
    double scale_X_privacy_budget = 0.4;
};

inline std::ostream& operator<<(std::ostream &os, const ModelParams &mp)
{
    return os << "ModelParams: " << std::endl
              << "    nb_trees: " << mp.nb_trees << std::endl
              << "    learning_rate: " << mp.learning_rate << std::endl
              << "    privacy_budget: " << mp.privacy_budget << std::endl
              << "    optimization_privacy_budget: " << mp.optimization_privacy_budget << std::endl
              << "    max_depth: " << mp.max_depth << std::endl
              << "    learning_rate: " << mp.learning_rate << std::endl
              << "    l2_lambda: " << mp.l2_lambda << std::endl
              << "    l2_threshold: " << mp.l2_threshold << std::endl
              << "    gradient_filtering: " << mp.gradient_filtering << std::endl
              << "    leaf_clipping: " << mp.leaf_clipping << std::endl;
}

// each tree has these additional parameters
struct TreeParams
{
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};

#endif /* PARAMETERS_H */

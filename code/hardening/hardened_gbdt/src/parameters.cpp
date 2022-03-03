#include <iostream>
#include "parameters.h"

ModelParams create_default_params()
{
    ModelParams params;
    params.nb_trees = 50;
    params.max_depth = 6;
    params.gradient_filtering = TRUE;
    params.balance_partition = TRUE;
    params.leaf_clipping = TRUE;
    params.privacy_budget = 0.1;
    return params;
};

std::ostream &operator<<(std::ostream &os, const ModelParams &mp)
{
    return os << "ModelParams: " << std::endl
              << "    nb_trees: " << mp.nb_trees << std::endl
              << "    learning_rate: " << mp.learning_rate << std::endl
              << "    privacy_budget: " << mp.privacy_budget << std::endl
              << "    optimization_privacy_budget: " << mp.optimization_privacy_budget << std::endl
              << "    max_depth: " << mp.max_depth << std::endl
              << "    l2_lambda: " << mp.l2_lambda << std::endl
              << "    l2_threshold: " << mp.l2_threshold << std::endl
              << "    gradient_filtering: " << mp.gradient_filtering << std::endl
              << "    leaf_clipping: " << mp.leaf_clipping << std::endl;
}
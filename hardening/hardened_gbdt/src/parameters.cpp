#include <iostream>
#include "parameters.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "tree_rejection.h"


std::ostream &operator<<(std::ostream &os, const ModelParams &mp)
{
    return os << "ModelParams: " << std::endl
              << "    n_trials: " << mp.n_trials << std::endl
              << "    n_trees_to_accept: " << mp.n_trees_to_accept << std::endl
              << "    learning_rate: " << mp.learning_rate << std::endl
              << "    privacy_budget: " << mp.privacy_budget << std::endl
              << "    ensemble_rejector_budget_split: " << mp.ensemble_rejector_budget_split << std::endl
              << "    tree_rejector: " << *mp.tree_rejector << std::endl
              << "    max_depth: " << mp.max_depth << std::endl
              << "    l2_lambda: " << mp.l2_lambda << std::endl
              << "    l2_threshold: " << mp.l2_threshold << std::endl
              << "    gradient_filtering: " << mp.gradient_filtering << std::endl
              << "    leaf_clipping: " << mp.leaf_clipping << std::endl;
}
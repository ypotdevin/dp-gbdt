#include <iostream>
#include "parameters.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "gbdt/tree_rejection.h"

std::ostream &operator<<(std::ostream &os, const ModelParams &mp)
{
    return os << "ModelParams: " << std::endl
              << "    n_trees_to_accept: " << mp.n_trees_to_accept << std::endl
              << "    learning_rate: " << mp.learning_rate << std::endl
              << "    privacy_budget: " << mp.privacy_budget << std::endl
              << "    training_variant: " << mp.training_variant << std::endl
              << "    ensemble_rejector_budget_split: " << mp.ensemble_rejector_budget_split << std::endl
              << "    dp_argmax_privacy_budget: " << mp.dp_argmax_privacy_budget << std::endl
              << "    dp_argmax_stopping_prob: " << mp.stopping_prob << std::endl
              << "    max_depth: " << mp.max_depth << std::endl
              << "    l2_lambda: " << mp.l2_lambda << std::endl
              << "    l2_threshold: " << mp.l2_threshold << std::endl
              << "    gradient_filtering: " << mp.gradient_filtering << std::endl
              << "    leaf_clipping: " << mp.leaf_clipping << std::endl;
}

std::ostream &operator<<(std::ostream &os, const TreeParams &tp)
{
    return os << "TreeParams: " << std::endl
              << "    delta_g: " << tp.delta_g << std::endl
              << "    delta_v: " << tp.delta_v << std::endl
              << "    tree_privacy_budget: " << tp.tree_privacy_budget << std::endl;
}
#include <iostream>
#include "parameters.h"
#include "cli_parser.h"
#include "logging.h"
#include "spdlog/spdlog.h"

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

void parse_model_parameters(cli_parser::CommandLineParser &cp, ModelParams &mp)
{
    if (cp.hasOption("--ensemble-privacy-budget"))
    {
        mp.privacy_budget = cp.getDoubleOptionValue("--ensemble-privacy-budget");
    }
    if (cp.hasOption("--optimization-privacy-budget"))
    {
        if (cp.hasOption("--no-optimization"))
        {
            LOG_INFO("conflicting command line options: --optimization-privacy-budget vs --no-optimization");
        }
        if (cp.hasOption("--leaky-optimization"))
        {
            LOG_INFO("conflicting command line options: --optimization-privacy-budget vs --leaky-optimization");
        }
        mp.optimization_privacy_budget = cp.getDoubleOptionValue("--optimization-privacy-budget");
        mp.optimize = true;
        mp.leaky_opt = false;
    }
    if (cp.hasOption("--gamma"))
    {
        mp.gamma = cp.getDoubleOptionValue("--gamma");
    }
    if (cp.hasOption("--nb-trees"))
    {
        mp.nb_trees = cp.getIntOptionValue("--nb-trees");
    }
    if (cp.hasOption("--max-depth"))
    {
        mp.max_depth = cp.getIntOptionValue("--max-depth");
    }
    if (cp.hasOption("--learning-rate"))
    {
        mp.learning_rate = cp.getDoubleOptionValue("--learning-rate");
    }
    if (cp.hasOption("--l2-lambda"))
    {
        mp.l2_lambda = cp.getDoubleOptionValue("--l2-lambda");
    }
    if (cp.hasOption("--l2-threshold"))
    {
        mp.l2_threshold = cp.getDoubleOptionValue("--l2-threshold");
    }
    if (cp.hasOption("--no-gradient-filtering"))
    {
        mp.gradient_filtering = FALSE;
    }
    else
    {
        mp.gradient_filtering = TRUE;
    }
    if (cp.hasOption("--no-leaf-clipping"))
    {
        mp.leaf_clipping = FALSE;
    }
    else
    {
        mp.leaf_clipping = TRUE;
    }
    if (cp.hasOption("--leaky-optimization"))
    {
        if (cp.hasOption("--no-optimization"))
        {
            LOG_INFO("conflicting command line options: --leaky-optimization vs --no-optimization");
        }
        mp.leaky_opt = true;
        mp.optimization_privacy_budget = std::nan("");
        mp.error_upper_bound = std::nan("");
    }
    else
    {
        mp.leaky_opt = false;
    }
    if (cp.hasOption("--no-optimization"))
    {
        if (cp.hasOption("--leaky-optimization"))
        {
            LOG_INFO("conflicting command line options: --no-optimization vs --leaky-optimization");
        }
        mp.optimize = false;
        mp.optimization_privacy_budget = std::nan("");
        mp.error_upper_bound = std::nan("");
    }
    else
    {
        mp.optimize = true;
    }
}

std::ostream &operator<<(std::ostream &os, const ModelParams &mp)
{
    return os << "ModelParams: " << std::endl
              << "    nb_trees: " << mp.nb_trees << std::endl
              << "    learning_rate: " << mp.learning_rate << std::endl
              << "    privacy_budget: " << mp.privacy_budget << std::endl
              << "    gamma: " << mp.gamma << std::endl
              << "    optimize:" << mp.optimize << std::endl
              << "    optimization_privacy_budget: " << mp.optimization_privacy_budget << std::endl
              << "    leaky-optimization:" << mp.leaky_opt << std::endl
              << "    max_depth: " << mp.max_depth << std::endl
              << "    l2_lambda: " << mp.l2_lambda << std::endl
              << "    l2_threshold: " << mp.l2_threshold << std::endl
              << "    gradient_filtering: " << mp.gradient_filtering << std::endl
              << "    leaf_clipping: " << mp.leaf_clipping << std::endl
              << "    error_upper_bound: " << mp.error_upper_bound << std::endl;
}
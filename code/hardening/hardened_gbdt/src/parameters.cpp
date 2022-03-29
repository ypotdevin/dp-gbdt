#include <iostream>
#include "parameters.h"
#include "cli_parser.h"

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

/**
 * @brief Update given model parameters by what is passed via command line.
 *
 * Currently accepted model parameters:
 *   --ensemble-privacy-budget (double)
 *   --optimization-privacy-budget (double)
 *   --nb-trees (int)
 *   --max-depth (int)
 *   --learning-rate (double)
 *   --l2-lambda (double)
 *   --l2-threshold (double)
 *   --no-gradient-filtering (boolean flag)
 *   --no-leaf-clipping (boolean flag)
 *
 * @param cp the parser holding the command line arguments.
 * @param mp the model parameters to update.
 */
void parse_model_parameters(cli_parser::CommandLineParser &cp, ModelParams &mp)
{
    if (cp.hasOption("--ensemble-privacy-budget"))
    {
        mp.privacy_budget = cp.getDoubleOptionValue("--ensemble-privacy-budget");
    }
    if (cp.hasOption("--optimization-privacy-budget"))
    {
        mp.optimization_privacy_budget = cp.getDoubleOptionValue("--optimization-privacy-budget");
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
        mp.leaky_opt = true;
    }
    else
    {
        mp.leaky_opt = false;
    }

}

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
#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory>
#include <vector>
#include "utils.h"
#include "loss.h"
#include "cli_parser.h"

struct ModelParams
{
    // primary model parameters
    int nb_trees = 50;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    double optimization_privacy_budget = 1.0;
    int max_depth = 6;
    int min_samples_split = 2;
    unsigned balance_partition = TRUE;
    unsigned gradient_filtering = TRUE;
    unsigned leaf_clipping = TRUE;
    unsigned use_decay = FALSE;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;

    // secondary model parameters
    unsigned use_grid = FALSE;
    std::tuple<double, double> grid_borders;
    double grid_step_size;
    std::vector<std::vector<double>> cat_values;
    unsigned scale_X = FALSE;
    double scale_X_percentile = 95;
    double scale_X_privacy_budget = 0.4;
    unsigned scale_y = FALSE;

    // dataset specific parameters
    std::shared_ptr<Task> task;
    double error_upper_bound = 10.0;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
};

// create some default parameters for quick testing
ModelParams create_default_params();

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
void parse_model_parameters(cli_parser::CommandLineParser &cp, ModelParams &mp);

std::ostream &operator<<(std::ostream &os, const ModelParams &mp);

// each tree has these additional parameters
struct TreeParams
{
    double delta_g;
    double delta_v;
    double tree_privacy_budget;
};

#endif /* PARAMETERS_H */

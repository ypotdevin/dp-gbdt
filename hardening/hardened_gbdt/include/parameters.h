#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory>
#include <random>
#include <vector>
#include "cli_parser.h"
#include "loss.h"
#include "utils.h"
#include "tree_rejection.h"

struct ModelParams
{
    // primary model parameters
    std::mt19937 rng;
    int nb_trees = 50;
    double learning_rate = 0.1;
    double privacy_budget = 1.0;
    std::shared_ptr<tree_rejection::TreeRejector> tree_rejector;
    int max_depth = 6;
    int min_samples_split = 2;
    unsigned balance_partition = true;
    unsigned gradient_filtering = true;
    unsigned leaf_clipping = true;
    unsigned use_decay = false;
    double l2_threshold = 1.0;
    double l2_lambda = 0.1;

    // secondary model parameters
    unsigned use_grid = false;
    std::tuple<double, double> grid_borders;
    double grid_step_size;
    std::vector<std::vector<double>> cat_values;
    unsigned scale_X = false;
    double scale_X_percentile = 95;
    double scale_X_privacy_budget = 0.4;
    unsigned scale_y = false;

    // dataset specific parameters
    std::shared_ptr<Task> task;
    std::vector<int> cat_idx;
    std::vector<int> num_idx;
};

/**
 * @brief Update given model parameters by what is passed via command line.
 *
 * Currently accepted model parameters:
 *   --ensemble-privacy-budget (double)
 *   --no-tree-rejection (selection flag)
 *   --quantile-rejection (selection flag)
 *   --quantile-rejection-q (double)
 *   --dp-rmse-tree-rejection (selection flag)
 *   --rejection-budget (double)
 *   --error-upper-bound (double)
 *   --dp-rmse-gamma (double)
 *   --nb-trees (int)
 *   --max-depth (int)
 *   --learning-rate (double)
 *   --l2-lambda (double)
 *   --l2-threshold (double)
 *   --seed (int)
 *   --no-gradient-filtering (boolean flag, default: active gradient filtering)
 *   --no-leaf-clipping (boolean flag, default: active leaf clipping)
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

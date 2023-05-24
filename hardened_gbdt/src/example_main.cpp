#include <iostream>

#include "estimator.h"
#include "spdlog/spdlog.h"
#include "gbdt/utils.h"

// only for diagnosis
#include "dataset_parser.h"
#include "parameters.h"

void vanilla()
{
    spdlog::set_level(spdlog::level::info);

    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(42);

    dpgbdt::Estimator regressor(
        rng,
        1.0,  // privacy_budget
        -1.0, // ensemble_rejector_budget_split (irrelevant)
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(20.0, 2.0, rng)),
        -1.0, // dp_argmax_privacy_budget (irrelevant)
        -1.0, // dp_argmax_stopping_prob (irrelevant)
        0.01, // learning_rate
        50,   // n_trees_to_accept
        6,    // max_depth
        2,    // min_samples_split
        5.63, // l2_threshold
        0.7,  // l2_lambda
        true,
        true,
        true,
        false,
        "warning");
    regressor.fit(dataset->X,
                  dataset->y,
                  parameters.cat_idx,
                  parameters.num_idx,
                  parameters.grid_lower_bounds,
                  parameters.grid_upper_bounds,
                  parameters.grid_step_sizes);
    auto y_pred = regressor.predict(dataset->X);
    auto rmse = compute_rmse(dataset->y, y_pred);
    std::cout << rmse << std::endl;
    delete dataset;
}

void nissim_rmse()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(42);

    dpgbdt::Estimator regressor(
        rng,
        1.0, // privacy_budget
        0.4, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(4.83, 2.0, rng)),
        1e-4, // dp_argmax_privacy_budget
        0.01, // dp_argmax_stopping_prob
        0.1,  // learning_rate
        5,    // n_trees_to_accept
        1,    // max_depth
        2,    // min_samples_split
        0.5,  // l2_threshold
        0.1,  // l2_lambda
        true,
        true,
        true,
        false,
        "warning");
    regressor.fit(dataset->X,
                  dataset->y,
                  parameters.cat_idx,
                  parameters.num_idx,
                  parameters.grid_lower_bounds,
                  parameters.grid_upper_bounds,
                  parameters.grid_step_sizes);
    auto y_pred = regressor.predict(dataset->X);
    auto rmse = compute_rmse(dataset->y, y_pred);
    std::cout << rmse << std::endl;
    delete dataset;
}

void bun_steinke_rmse()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(42);

    dpgbdt::Estimator regressor(
        rng,
        1.0,  // privacy_budget
        0.75, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::BunSteinkeScorer>(new tree_rejection::BunSteinkeScorer(7.0, 7.29 * 1e-4, 1e-6, rng)),
        0.001, // dp_argmax_privacy_budget
        0.2,   // dp_argmax_stopping_prob
        0.1,   // learning_rate
        5,     // n_trees_to_accept
        6,     // max_depth
        2,     // min_samples_split
        0.5,   // l2_threshold
        0.1,   // l2_lambda
        true,
        true,
        true,
        false,
        "warning");
    regressor.fit(dataset->X,
                  dataset->y,
                  parameters.cat_idx,
                  parameters.num_idx,
                  parameters.grid_lower_bounds,
                  parameters.grid_upper_bounds,
                  parameters.grid_step_sizes);
    auto y_pred = regressor.predict(dataset->X);
    auto rmse = compute_rmse(dataset->y, y_pred);
    std::cout << rmse << std::endl;
    delete dataset;
}

void privacy_buckets_rmse()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(42);

    dpgbdt::Estimator regressor(
        rng,
        1.0, // privacy_budget
        0.2, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::PrivacyBucketScorer>(
            new tree_rejection::PrivacyBucketScorer(
                7.0,
                729 * 1e-6,
                5,
                {0.0, 1.25, -0.0361, 0.0, 0.0, -2.10, 0.0, 0.0, 0.0},
                rng)),
        0.01, // dp_argmax_privacy_budget
        0.01, // dp_argmax_stopping_prob
        0.1,  // learning_rate
        5,    // n_trees_to_accept
        1,    // max_depth
        2,    // min_samples_split
        0.5,  // l2_threshold
        0.1,  // l2_lambda
        true,
        true,
        true,
        false,
        "warning");
    regressor.fit(dataset->X,
                  dataset->y,
                  parameters.cat_idx,
                  parameters.num_idx,
                  parameters.grid_lower_bounds,
                  parameters.grid_upper_bounds,
                  parameters.grid_step_sizes);
    auto y_pred = regressor.predict(dataset->X);
    auto rmse = compute_rmse(dataset->y, y_pred);
    std::cout << rmse << std::endl;
    delete dataset;
}

int main(int argc, char **argv)
{
    bun_steinke_rmse();
}
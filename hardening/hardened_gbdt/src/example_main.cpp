#include <iostream>

#include "estimator.h"
#include "spdlog/spdlog.h"
#include "gbdt/utils.h"

// only for diagnosis
#include "dataset_parser.h"
#include "parameters.h"

void basic_run()
{
    spdlog::set_level(spdlog::level::info);

    std::vector<std::vector<double>> X{
        {0.0, -2.3, 0.11},
        {1.0, 1.4, 0.52},
        {0.0, 6.0, 0.33},
        {2.0, 0.0, 0.99},
        {1.0, -7.2, 0.24}};
    std::vector<double> y{1.0, 4.0, 2.0, 2.9, 1.0};
    std::vector<int> cat_idx{0};
    std::vector<int> num_idx{1, 2, 3};
    std::vector<double> grid_lower_bounds = {0.0, -10.0, 0.0};
    std::vector<double> grid_upper_bounds = {2.0, 10.0, 1.0};
    std::vector<double> grid_step_sizes = {1.0, 1.0, 0.01};

    dpgbdt::Estimator regressor;
    regressor.fit(X,
                  y,
                  cat_idx,
                  num_idx,
                  grid_lower_bounds,
                  grid_upper_bounds,
                  grid_step_sizes);
    auto y_pred = regressor.predict(X);
    auto rmse = compute_rmse(y, y_pred);
    std::cout << rmse << std::endl;
}

void diagnosis_run()
{
    spdlog::set_level(spdlog::level::debug);

    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 50);

    dpgbdt::Estimator regressor;
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

void good_vanilla_setting()
{
    spdlog::set_level(spdlog::level::info);

    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(std::random_device{}());

    dpgbdt::Estimator regressor(
        // 'learning_rate': 1.5612896582519962, 'max_depth': 4.859616522088254,
        // 'l2_threshold': 2.187546396545948, 'l2_lambda': 42.34056387330428,
        // 'n_trees_to_accept': 1.003062181872044, 'privacy_budget': 1.0,
        // 'ensemble_rejector_budget_split': 1.0
        rng,
        1.0,
        1.0, // this will be ignored
        "vanilla",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(100.0, 2.0, rng)),        // this will be ignored
        0.1,                                                                                                     // this will be ignored
        0.05,                                                                                                    // this will be ignored
        1.561,                                                                                                   // learning_rate
        1,
        4,
        2,
        2.188,
        42.341,
        true,
        true,
        true,
        false,
        "debug");
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

void good_dp_argmax_rmse_scoring_setting()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(std::random_device{}());

    dpgbdt::Estimator regressor(
        rng,
        10.0, // privacy_budget
        0.90, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(20.0, 2.0, rng)),
        0.001, // dp_argmax_privacy_budget
        0.01,  // dp_argmax_stopping_prob
        0.1,   // learning_rate
        50,    // n_trees_to_accept
        6,     // max_depth
        2,     // min_samples_split
        9.167, // l2_threshold
        0.1,   // l2_lambda
        true,
        true,
        true,
        false,
        "debug");
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

void long_dp_argmax_rmse_scoring_setting()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(std::random_device{}());

    dpgbdt::Estimator regressor(
        rng,
        10.0, // privacy_budget
        0.6,  // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(20.0, 2.0, rng)),
        0.01, // dp_argmax_privacy_budget
        0.1,  // dp_argmax_stopping_prob
        0.1,  // learning_rate
        50,   // n_trees_to_accept
        6,    // max_depth
        2,    // min_samples_split
        0.5,  // l2_threshold
        0.1,  // l2_lambda
        true,
        true,
        true,
        false,
        "info");
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

void bun_steinke()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(std::random_device{}());

    dpgbdt::Estimator regressor(
        rng,
        1.0,  // privacy_budget
        0.70, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::BunSteinkeScorer>(new tree_rejection::BunSteinkeScorer(20.0, 3.14, 1e-6, rng)),
        0.001, // dp_argmax_privacy_budget
        0.05,  // dp_argmax_stopping_prob
        0.1,   // learning_rate
        5,     // n_trees_to_accept
        6,     // max_depth
        2,     // min_samples_split
        9.167, // l2_threshold
        0.1,   // l2_lambda
        true,
        true,
        true,
        false,
        "debug");
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

void dp_rmse2()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(42);
    std::shared_ptr<tree_rejection::Beta> beta_ptr(
        std::shared_ptr<tree_rejection::ConstantBeta>(
            new tree_rejection::ConstantBeta(42.0)));

    dpgbdt::Estimator regressor(
        rng,
        1.0,  // privacy_budget
        0.70, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::DPrMSEScorer2>(new tree_rejection::DPrMSEScorer2(beta_ptr, 20.0, 2.0, rng)),
        0.001, // dp_argmax_privacy_budget
        0.05,  // dp_argmax_stopping_prob
        0.1,   // learning_rate
        5,     // n_trees_to_accept
        6,     // max_depth
        2,     // min_samples_split
        9.167, // l2_threshold
        0.1,   // l2_lambda
        true,
        true,
        true,
        false,
        "debug");
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

void privacy_buckets()
{
    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(std::random_device{}());

    dpgbdt::Estimator regressor(
        rng,
        1.0,  // privacy_budget
        0.70, // ensemble_rejector_budget_split
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::PrivacyBucketScorer>(
            new tree_rejection::PrivacyBucketScorer(
                20.0,
                729 * 1e-6,
                5,
                {0.0, 1.25, -0.0361, 0.0, 0.0, -2.10, 0.0, 0.0, 0.0},
                rng)),
        0.001, // dp_argmax_privacy_budget
        0.05,  // dp_argmax_stopping_prob
        0.1,   // learning_rate
        5,     // n_trees_to_accept
        6,     // max_depth
        2,     // min_samples_split
        9.167, // l2_threshold
        0.1,   // l2_lambda
        true,
        true,
        true,
        false,
        "debug");
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
    // good_dp_argmax_rmse_scoring_setting();
    // bun_steinke();
    // dp_rmse2();
    privacy_buckets();
}
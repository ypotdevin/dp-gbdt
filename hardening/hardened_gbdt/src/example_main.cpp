#include <iostream>

#include "estimator.h"
#include "spdlog/spdlog.h"
#include "utils.h"

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

    dpgbdt::Estimator regressor;
    regressor.fit(X, y, cat_idx, num_idx);
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
    regressor.fit(dataset->X, dataset->y, parameters.cat_idx, parameters.num_idx);
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
        1.561,
        1,
        4,
        2,
        2.188,
        42.341,
        true,
        true,
        true,
        false);
    regressor.fit(dataset->X, dataset->y, parameters.cat_idx, parameters.num_idx);
    auto y_pred = regressor.predict(dataset->X);
    auto rmse = compute_rmse(dataset->y, y_pred);
    std::cout << rmse << std::endl;
    delete dataset;
}

void good_dp_argmax_scoring_setting()
{
    // TODO: These settings have to be updated by measured ones
    // So far, these are just a guess
    spdlog::set_level(spdlog::level::debug);

    ModelParams parameters;
    DataSet *dataset = Parser::get_abalone(parameters, 5000);

    std::mt19937 rng(std::random_device{}());

    dpgbdt::Estimator regressor(
        // 'learning_rate': 1.5612896582519962, 'max_depth': 4.859616522088254,
        // 'l2_threshold': 2.187546396545948, 'l2_lambda': 42.34056387330428,
        // 'n_trees_to_accept': 1.003062181872044, 'privacy_budget': 1.0,
        // 'ensemble_rejector_budget_split': 1.0
        rng,
        100.0,
        0.02,
        "dp_argmax_scoring",
        std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, rng)), // this will be ignored
        std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(100.0, 2.0, rng)),
        10.0,
        0.05,
        1.561,
        1,
        4,
        2,
        2.188,
        42.341,
        true,
        true,
        true,
        false);
    regressor.fit(dataset->X, dataset->y, parameters.cat_idx, parameters.num_idx);
    auto y_pred = regressor.predict(dataset->X);
    auto rmse = compute_rmse(dataset->y, y_pred);
    std::cout << rmse << std::endl;
    delete dataset;
}

int main(int argc, char **argv)
{
    good_dp_argmax_scoring_setting();
}
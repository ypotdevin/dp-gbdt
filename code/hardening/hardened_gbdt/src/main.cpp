#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "logging.h"
#include "spdlog/spdlog.h"
#include "utils.h"
#include "parameters.h"
#include "gbdt/dp_ensemble.h"
#include "dataset_parser.h"
#include "data.h"
#include "cli_parser.h"
#include "evaluation.h"

spdlog::level::level_enum select_log_level(std::string level)
{
    if (level == "off")
    {
        return spdlog::level::off;
    }
    else if (level == "error")
    {
        return spdlog::level::err;
    }
    else if (level == "info")
    {
        return spdlog::level::info;
    }
    else if (level == "debug")
    {
        return spdlog::level::debug;
    }
    else
    {
        throw std::runtime_error("Unknown log level");
    }
}

int main(int argc, char **argv)
{
    cli_parser::CommandLineParser cp(argc, argv);
    int seed;
    std::mt19937_64 rng;
    if (cp.hasOption("--seed"))
    {
        seed = cp.getIntOptionValue("--seed");
        rng = std::mt19937_64(seed);
    }
    else
    {
        throw std::runtime_error("Parameter seed is missing.");
    }

    if (cp.hasOption("--log-level"))
    {
        spdlog::set_level(select_log_level(cp.getOptionValue("--log-level")));
    }
    else
    {
        spdlog::set_level(select_log_level("off"));
    }
    spdlog::set_pattern("[%H:%M:%S] [%^%5l%$] %v");

    // Define model parameters
    // reason to use a vector is because parser expects it
    ModelParams params;
    parse_model_parameters(cp, params);
    DataSet *dataset = parse_dataset_parameters(cp, params);

    // create cross validation inputs
    int num_folds = 5; // the k in k-fold
    std::vector<TrainTestSplit *> cv_inputs = create_cross_validation_inputs(dataset, num_folds);
    delete dataset;

    std::vector<double> train_scores, test_scores;
    // do cross validation
    for (auto split : cv_inputs)
    {
        DPEnsemble ensemble = DPEnsemble(&params, rng);
        ensemble.train(&split->train);

        // predict with the test set
        std::vector<double> y_train_pred = ensemble.predict(split->train.X);
        std::vector<double> y_test_pred = ensemble.predict(split->test.X);

        if (is_true(params.scale_y))
        {
            inverse_scale_y(params, split->train.scaler, y_train_pred);
        }
        if (is_true(params.scale_y))
        {
            inverse_scale_y(params, split->train.scaler, y_test_pred);
        }

        // compute score
        double train_score = params.task->compute_score(split->train.y, y_train_pred);
        train_scores.push_back(train_score);
        double test_score = params.task->compute_score(split->test.y, y_test_pred);
        test_scores.push_back(test_score);

        delete split;
    }
    if (cp.hasOption("--results-file"))
    {
        auto filename = cp.getOptionValue("--results-file");
        evaluation::write_csv_file(filename, params, "rmse", train_scores, test_scores, seed);
    }
}
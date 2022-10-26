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

int main(int argc, char **argv)
{
    diagnosis_run();
}
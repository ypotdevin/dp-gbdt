#include <iostream>

#include "estimator.h"
#include "spdlog/spdlog.h"

int main(int argc, char **argv)
{
    spdlog::set_level(spdlog::level::info);

    dpgbdt::Estimator regressor;

    std::vector<std::vector<double>> X{
        {0.0, -2.3, 0.11},
        {1.0, 1.4, 0.52},
        {0.0, 6.0, 0.33},
        {2.0, 0.0, 0.99},
        {1.0, -7.2, 0.24}};
    std::vector<double> y{1.0, 4.0, 2.0, 2.9, 1.0};
    std::vector<int> cat_idx{0};
    std::vector<int> num_idx{1, 2, 3};

    regressor.fit(X, y, cat_idx, num_idx);
    auto y_pred = regressor.predict(X);
    std::transform(y_pred.begin(), y_pred.end(),
                   y.begin(), y_pred.begin(), std::minus<double>());
    auto rmse = compute_rmse(y_pred);
    std::cout << rmse << std::endl;
}
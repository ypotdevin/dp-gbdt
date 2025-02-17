#include <algorithm>
#include <cmath>
#include <numeric>
#include <mutex>
#include <random>
#include "gbdt/utils.h"
#include "constant_time.h"

/** Methods */

// put a value between two bounds, not in std::algorithm in c++11
double clamp(double n, double lower, double upper)
{
    n = constant_time::select(n < lower, lower, n);
    n = constant_time::select(n > upper, upper, n);
    return n;
}

void clamp(std::vector<double> &to_clamp, double lower, double upper)
{
    std::transform(to_clamp.begin(), to_clamp.end(), to_clamp.begin(), [lower, upper](double e)
                   { return clamp(e, lower, upper); });
}

double log_sum_exp(std::vector<double> vec)
{
    size_t count = vec.size();
    if (count > 0)
    {
        double max_val = std::numeric_limits<double>::min();
        for (auto elem : vec)
        {
            max_val = constant_time::max(max_val, elem);
        }
        double sum = 0;
        for (size_t i = 0; i < count; i++)
        {
            sum += exp(vec[i] - max_val);
        }
        return log(sum) + max_val;
    }
    else
    {
        return 0.0;
    }
}

std::vector<double> absolute_differences(const std::vector<double> &source, const std::vector<double> &target)
{
    std::vector<double> abs_errors(source.size());
    std::transform(source.begin(), source.end(),
                   target.begin(), abs_errors.begin(), [](double s, double t)
                   { return std::abs(s - t); });
    return abs_errors;
}

double compute_mean(std::vector<double> &vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}

double compute_stddev(std::vector<double> &vec, double mean)
{
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    return std::sqrt(sq_sum / vec.size() - mean * mean);
}

double compute_rmse(std::vector<double> differences)
{
    transform(differences.begin(), differences.end(), differences.begin(), [](double x)
              { return x * x; });
    auto mean = compute_mean(differences);
    auto rmse = std::sqrt(mean);
    return rmse;
}

double compute_rmse(const std::vector<double> &source, const std::vector<double> &target)
{
    auto abs_errors = absolute_differences(source, target);
    return compute_rmse(abs_errors);
}

std::string get_time_string()
{
    time_t t = time(0);
    struct tm *now = localtime(&t);
    char buffer[80];
    strftime(buffer, 80, "%m.%d_%H:%M", now);
    return std::string(buffer);
}

void normalize(std::vector<double> &values)
{
    auto normalization_factor = std::accumulate(values.begin(), values.end(), 0.0);
    std::transform(values.begin(), values.end(), values.begin(), [normalization_factor](double value)
                   { return value / normalization_factor; });
}

double poly2d(double x, double y, const std::vector<double> &coefficients)
{
    int degree = std::sqrt(coefficients.size());
    size_t idx = 0;
    double sum = 0.0;
    for (int j = 0; j < degree; ++j)
    {
        for (int i = 0; i < degree; i++)
        {
            sum += coefficients[idx] * std::pow(x, i) * std::pow(y, j);
            idx++;
        }
    }
    return sum;
}

namespace numpy
{
    std::size_t choice(const std::vector<double> &probabilities, std::mt19937 &rng)
    {
        std::discrete_distribution<std::size_t> index_distribution(probabilities.begin(), probabilities.end());
        return index_distribution(rng);
    }

    std::vector<double> linspace(double low, double high, double step_size)
    {
        if (low > high)
        {
            return std::vector<double>{};
        }
        else if (low == high)
        {
            return std::vector<double>{low};
        }
        else
        {
            size_t n_points = (high - low) / step_size + 1;
            std::vector<double> points(n_points);

            for (size_t i = 0; i < n_points; ++i)
            {
                points[i] = low + i * step_size;
            }
            return points;
        }
    }
}

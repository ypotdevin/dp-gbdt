#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <string>
typedef std::vector<std::vector<double>> VVD;

// method declarations
double clamp(double n, double lower, double upper);
void clamp(std::vector<double> &to_clamp, double lower, double upper);
double log_sum_exp(std::vector<double> arr);
void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');
std::vector<double> absolute_differences(const std::vector<double> &source, const std::vector<double> &target);
double compute_mean(std::vector<double> &vec);
double compute_stddev(std::vector<double> &vec, double mean);
double compute_rmse(std::vector<double> differences);
double compute_rmse(const std::vector<double> &source, const std::vector<double> &target);
std::string get_time_string();
std::vector<double> linspace(double low, double high, size_t num);

void normalize(std::vector<double> &values);
/**
 * @brief Evaluate a two-dimensional polynomial determined by coefficients at
 * x and y.
 *
 * @param x point in first dimension
 * @param y point in second dimension
 * @param coefficients the coefficients map to
 * x^0*y^0, x^1*y^0, ..., x^d*y^0, x^0*y^1, x^1*y^1, ..., x^d*y_1, ..., x^d*y^d
 * (exponent of x changes fast, exponent of y changes slowly)
 * @return double poly(x, y)
 */
double poly2d(double x, double y, const std::vector<double> &coefficients);

namespace numpy
{
    std::size_t choice(const std::vector<double> &probabilities, std::mt19937 &rng);
    std::vector<double> linspace(double low, double high, double step_size);
}

#endif // UTILS_H
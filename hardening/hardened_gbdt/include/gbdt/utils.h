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
/**
 * @brief
 *
 * @param source
 * @param target
 * @return std::vector<double>
 */
std::vector<double> absolute_differences(const std::vector<double> &source, const std::vector<double> &target);
double compute_mean(std::vector<double> &vec);
double compute_stdev(std::vector<double> &vec, double mean);
double compute_rmse(std::vector<double> differences);
double compute_rmse(const std::vector<double> &source, const std::vector<double> &target);
std::string get_time_string();
std::vector<double> linspace(double low, double high, size_t num);
void normalize(std::vector<double> &values);

namespace numpy
{
    std::size_t choice(const std::vector<double> &probabilities, std::mt19937 &rng);
}

#endif // UTILS_H
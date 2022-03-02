#ifndef UTILS_H
#define UTILS_H

// use these numbers instead of bools. Their hamming distance is 26,
// which makes a fault injection attack on the parameters pretty much impossible.
// HAMMING_TRUE:  01101100001011110101111000011011
// HAMMING_FALSE: 00000001110001101010000111100100
#define HAMMING_TRUE 1815043611u
#define HAMMING_FALSE 29794788u

#include <vector>
#include <string>
#include "parameters.h"
#include "data.h"

namespace utils
{
    typedef std::vector<std::vector<double>> VVD;
    // method declarations
    ModelParams create_default_params();
    double clamp(double n, double lower, double upper);
    double log_sum_exp(std::vector<double> arr);
    void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');
    double compute_mean(std::vector<double> &vec);
    double compute_stdev(std::vector<double> &vec, double mean);
    std::string get_time_string();
    bool is_true(unsigned value);
}

#endif // UTILS_H
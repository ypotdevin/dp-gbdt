#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
typedef std::vector<std::vector<double>> VVD;
//#include "parameters.h"
//#include "data.h"



// method declarations
double clamp(double n, double lower, double upper);
double log_sum_exp(std::vector<double> arr);
void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');
double compute_mean(std::vector<double> &vec);
double compute_stdev(std::vector<double> &vec, double mean);
double compute_rmse(std::vector<double> differences);
std::string get_time_string();
std::vector<double> linspace(double low, double high, size_t num);



#endif // UTILS_H
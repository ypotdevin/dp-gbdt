#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <set>
#include <algorithm>
#include <random>
#include <ctime>
#include <stdexcept>
#include <string>
#include <iostream>
#include <iomanip> 
#include <sstream>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>
#include <iterator>


/* Logging, to be removed */
#define concat(one, two) ((std::string) one + (std::string) two).c_str()
#define LOG_INFO_ARG(msg, ...) spdlog::info(concat("[{0:>17s}] ", msg), __func__,  __VA_ARGS__)
#define LOG_INFO_NO_ARG(msg) spdlog::info(concat("[{0:>17s}] ", msg), __func__)
#define LOG_DEBUG_ARG(msg, ...) spdlog::debug(concat("[{0:>17s}] ", msg), __func__,  __VA_ARGS__)
#define LOG_DEBUG_NO_ARG(msg) spdlog::debug(concat("[{0:>17s}] ", msg), __func__)
#define GET_9TH_ARG(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, ...) arg9
#define LOG_INFO_MACRO_CHOOSER(...) \
    GET_9TH_ARG(__VA_ARGS__, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_ARG, LOG_INFO_NO_ARG, )
#define LOG_INFO(...) LOG_INFO_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define LOG_DEBUG_MACRO_CHOOSER(...) \
    GET_9TH_ARG(__VA_ARGS__, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_ARG, LOG_DEBUG_NO_ARG, )
#define LOG_DEBUG(...) LOG_DEBUG_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)
#define BOLD(words) "\033[0;40;33m" + words + "\033[0m"


typedef std::vector<std::vector<float>> VVF;

static const bool RANDOMIZATION = true;

using namespace std;

struct ModelParams {
    int nb_trees;
    float learning_rate = 0.1;
    float privacy_budget = 1.0;
    //LossFunction loss;
    int max_depth = 6;
    int early_stop = 5;
    int max_leaves;
    int min_samples_split = 2;
    bool second_split = true;
    bool balance_partition = true;
    bool gradient_filtering = false;
    bool leaf_clipping = false;
    bool use_dp = false; // TODO remove this default
    bool use_dfs = true;
    bool use_3_trees = false;
    bool use_decay = false;
    int test_size = 0.3;    // TODO 1st or 2nd split?
    int verbosity = -1;
    float l2_threshold = 1.0;
    float l2_lambda = 0.1;

    float init_score;
    vector<int> cat_idx;
    vector<int> num_idx;
};


struct Scaler {
    float data_min, data_max;
    float feature_min, feature_max;
    float scale, min_;
    Scaler() {};
    Scaler(float min_val, float max_val, float fmin, float fmax);
};

struct DataSet {
    VVF X;
    vector<float> y;
    vector<float> gradients;
    int length;
    int num_x_cols;
    bool empty;
    Scaler scaler;

    DataSet();
    DataSet(VVF X, vector<float> y);

    void add_row(vector<float> xrow, float yval); // probably broken because theres no self/this
    void scale(float lower, float upper);
};

struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
};

struct SplitCandidate {
    int feature_index;
    float split_value;
    float gain;
    SplitCandidate(int f, float s, float g) : feature_index(f), split_value(s), gain(g) {};
};

struct TreeParams {
    float delta_g;
    double delta_v;
    double tree_privacy_budget;
};

void inverse_scale(Scaler &scaler, vector<float> &vec);
float clip(float n, float lower, float upper);
vector<string> split_string(const string &s, char delim);           // TODO enable shuffle
TrainTestSplit train_test_split_random(DataSet dataset, float train_ratio = 0.70, bool shuffle = false);

double log_sum_exp(vector<double> arr);
void string_pad(std::string &str, const size_t num, const char paddingChar = ' ');

class Laplace
{
private:
    float scale;
    std::mt19937 generator;
    std::default_random_engine generator1;
    std::default_random_engine generator2;
    std::exponential_distribution<float> distribution;
public:
    Laplace(int seed): generator(seed){};
    Laplace(float scale, int seed): scale(scale), generator(seed), distribution(1.0/scale){};
    float return_a_random_variable();
    float return_a_random_variable(float scale);
};

#endif // UTILS_H
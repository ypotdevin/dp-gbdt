#include "utils.h"


/** Global Variables */
bool VERIFICATION_MODE;
bool RANDOMIZATION;
size_t cv_fold_index;


// struct ModelParams{
//     int nb_trees;
//     double learning_rate = 0.1;
//     double privacy_budget = 1.0;
//     LossFunction lossfunction;
//     int max_depth = 6;
//     int early_stop = 5;
//     int max_leaves;
//     int min_samples_split = 2;
//     bool second_split = true;
//     bool balance_partition = true;
//     bool gradient_filtering = false;
//     bool leaf_clipping = false;
//     bool use_dp = false; // TODO remove this default
//     bool use_dfs = true;
//     bool use_3_trees = false;
//     bool use_decay = false;
//     int test_size = 0.3;    // TODO 1st or 2nd split?
//     int verbosity = -1;
//     double l2_threshold = 1.0;
//     double l2_lambda = 0.1;

//     double init_score;
//     vector<int> cat_idx;
//     vector<int> num_idx;
// };


vector<string> split_string(const string &s, char delim)
{
    vector<string> result;
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}


double clip(double n, double lower, double upper)
{
  return std::max(lower, std::min(n, upper));
}


std::string string_pad(std::string str, const size_t num, const char paddingChar)
{
    if(num > str.size())
        str.insert(0, num - str.size(), paddingChar);
    return str;
}


// TODO does this overflow?
double log_sum_exp(vector<double> vec)
{
    size_t count = vec.size();
    if (count > 0) {
        double maxVal = *max_element(vec.begin(), vec.end());
        double sum = 0;
        for (size_t i = 0; i < count; i++) {
            sum += exp(vec[i] - maxVal);
        }
        return log(sum) + maxVal;
    } else {
        return 0.0;
    }
}


ModelParams create_default_params()
{
    ModelParams params;
    params.nb_trees = 50;
    params.max_depth = 6;
    params.gradient_filtering = true;
    params.leaf_clipping = true;
    params.privacy_budget = 0.1;
    return params;
}


Scaler::Scaler(double min_val, double max_val, double fmin, double fmax) : data_min(min_val), data_max(max_val),
        feature_min(fmin), feature_max(fmax)
{
    double data_range = data_max - data_min;
    data_range = data_range == 0 ? 1 : data_range;
    this->scale = (this->feature_max - this->feature_min) / data_range;
    this->min_ = this->feature_min - this->data_min * this->scale;
}

DataSet::DataSet()
{
    empty = true;
}

DataSet::DataSet(VVD X, vector<double> y) : X(X), y(y)
{
    if(X.size() != y.size()){
        stringstream message;
        message << "X,y need equal amount of rows! (" << X.size() << ',' << y.size() << ')';
        throw runtime_error(message.str());
    }
    length = X.size();
    num_x_cols = X[0].size();
    empty = false;
}


void DataSet::add_row(vector<double> xrow, double yval)
{
    this->X.push_back(xrow);
    this->y.push_back(yval);
    length++;
}


// scale the features such that they lie in [lower,upper]
// !!! Seems like only y needs to be scaled !!!
void DataSet::scale(double lower, double upper)
{
    double doublemax = numeric_limits<double>::max();
    double doublemin = numeric_limits<double>::min();
    // vector<double> minima_x(num_x_cols,doublemax), maxima_x(num_x_cols,doublemin);
    double minimum_y = doublemax, maximum_y = doublemin;
    for(int i=0; i<length; i++) {
        // for(int j=0; j<num_x_cols; j++) {
        //     minima_x[j] = std::min(minima_x[j], X[i][j]);
        //     maxima_x[j] = std::max(maxima_x[j], X[i][j]);
        // }
        minimum_y = std::min(minimum_y, this->y[i]);
        maximum_y = std::max(maximum_y, this->y[i]);
    }
    for(int i=0; i<length; i++) {
        // only scale numerical features of X
        // for(auto j : params.num_idx) {
        //     X[i][j] = (X[i][j]-minima_x[j])/(maxima_x[j]-minima_x[j]) * (upper-lower) + lower; 
        // }
        // scale y as well
        this->y[i] = (this->y[i]- minimum_y)/(maximum_y-minimum_y) * (upper-lower) + lower;
    }
    this->scaler = Scaler(minimum_y, maximum_y, lower, upper);
}

void inverse_scale(Scaler &scaler, vector<double> &vec)
{
    // for(auto &elem : vec) {
    //     elem = (elem - scaler.feature_min) * (scaler.data_max - scaler.data_min)
    //                 / (scaler.feature_max - scaler.feature_min) + scaler.feature_min;
    // }

    // try 2
    // double upper = scaler.data_max;
    // double lower = scaler.data_min;
    // double minimum_y = numeric_limits<double>::max();
    // double maximum_y = numeric_limits<double>::min();
    // for(int i=0; i<vec.size(); i++) {
    //     // for(int j=0; j<num_x_cols; j++) {
    //     //     minima_x[j] = std::min(minima_x[j], X[i][j]);
    //     //     maxima_x[j] = std::max(maxima_x[j], X[i][j]);
    //     // }
    //     minimum_y = std::min(minimum_y, vec[i]);
    //     maximum_y = std::max(maximum_y, vec[i]);
    // }
    // for(int i=0; i<vec.size(); i++) {
    //     vec[i] = (vec[i]- minimum_y)/(maximum_y-minimum_y) * (upper-lower) + lower;
    // }
    for(auto &elem : vec) {
        elem -= scaler.min_;
        elem /= scaler.scale;
    }

}

TrainTestSplit train_test_split_random(DataSet dataset, double train_ratio, bool shuffle)
{
    if(shuffle) {
        srand(time(0));
        random_shuffle(dataset.X.begin(), dataset.X.end());
        random_shuffle(dataset.y.begin(), dataset.y.end());
    }

    // [ test |      train      ]
    int border = ceil((1-train_ratio) * dataset.y.size());

    VVD x_test(dataset.X.begin(), dataset.X.begin() + border);
    vector<double> y_test(dataset.y.begin(), dataset.y.begin() + border);
    VVD x_train(dataset.X.begin() + border, dataset.X.end());
    vector<double> y_train(dataset.y.begin() + border, dataset.y.end());

    if(train_ratio >= 1) {
        DataSet train(x_train, y_train);
        return TrainTestSplit(train, DataSet());
    } else if (train_ratio <= 0) {
        DataSet test(x_test, y_test);
        return TrainTestSplit(DataSet(), test);
    } else {
        DataSet train(x_train, y_train);
        DataSet test(x_test, y_test);
        return TrainTestSplit(train, test);
    }
}

vector<TrainTestSplit> create_cross_validation_inputs(DataSet &dataset, int folds, bool shuffle)
{
    if(shuffle) {
        srand(time(0));
        random_shuffle(dataset.X.begin(), dataset.X.end());
        random_shuffle(dataset.y.begin(), dataset.y.end());
    }

    int fold_size = dataset.length / folds;
    vector<int> fold_sizes(folds, fold_size);
    int remainder = dataset.length % folds;
    int index = 0;
    while (remainder != 0) {
        fold_sizes[index++]++;
        remainder--;
    }
    // each entry marks the first element of a fold (to be used as test set at some point)
    deque<int> indices(folds);
    std::partial_sum(fold_sizes.begin(), fold_sizes.end(), indices.begin());
    indices.push_front(0); 
    indices.pop_back();

    vector<TrainTestSplit> splits;

    for(int i=0; i<folds; i++) {
        VVD X_copy = dataset.X;
        vector<double> y_copy = dataset.y;

        VVD::iterator x_iterator = X_copy.begin() + indices[i];
        vector<double>::iterator y_iterator = y_copy.begin() + indices[i];

        VVD x_test(x_iterator, x_iterator + fold_sizes[i]);
        vector<double> y_test(y_iterator, y_iterator + fold_sizes[i]);

        X_copy.erase(x_iterator, x_iterator + fold_sizes[i]);
        y_copy.erase(y_iterator, y_iterator + fold_sizes[i]);

        VVD x_train(X_copy.begin(), X_copy.end());
        vector<double> y_train(y_copy.begin(), y_copy.end());

        DataSet train(x_train,y_train);
        DataSet test(x_test, y_test);

        splits.push_back(TrainTestSplit(train,test));
    }
    return splits;
}

double Laplace::return_a_random_variable(){
    double e1 = distribution(generator);
    double e2 = distribution(generator);
    return e1-e2;
}

double Laplace::return_a_random_variable(double scale){
    std::exponential_distribution<double> distribution1(1.0/scale);
    std::exponential_distribution<double> distribution2(1.0/scale);
    double e1 = distribution1(generator);
    double e2 = distribution2(generator);
    return e1-e2;
}


// extern std::ofstream verification_logfile;

// void VERIFICATION_LOG(...)
// {
//    verification_logfile << fmt::format(__VA_ARGS__) << "\n"; verification_logfile.flush();
// } 
  
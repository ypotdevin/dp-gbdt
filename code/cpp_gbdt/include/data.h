#ifndef DATA_H
#define DATA_H

#include <vector>
#include "utils.h"

// if the target needs to be scaled (into [-1,1]) before training, we store
// everything in this struct, that is required to invert the scaling after training 
struct Scaler {
    double data_min, data_max;
    double feature_min, feature_max;
    double scale, min_;
    bool scaling_required;
    Scaler() {};
    Scaler(double min_val, double max_val, double fmin, double fmax, bool scaling_required);
};

// basic wrapper around our data:
//  - matrix X
//  - target y
//  - space to store the gradients that are always updated
//  - some useful attributes
struct DataSet {
    VVD X;
    std::vector<double> y;
    std::vector<double> gradients;
    int length;
    int num_x_cols;
    bool empty;
    Scaler scaler;
    std::string name;
    std::string task;

    DataSet();
    DataSet(VVD X, std::vector<double> y);

    void add_row(std::vector<double> xrow, double yval);
    void scale(double lower, double upper);
};

// wrapper around 2 DataSets that belong together
struct TrainTestSplit {
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test) {};
};


// method declarations
void inverse_scale(Scaler &scaler, std::vector<double> &vec);
TrainTestSplit train_test_split_random(DataSet dataset, double train_ratio = 0.70, bool shuffle = false);
std::vector<TrainTestSplit> create_cross_validation_inputs(DataSet &dataset, int folds);


#endif /* DATA_H */

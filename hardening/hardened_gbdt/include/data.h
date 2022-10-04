#ifndef DATA_H
#define DATA_H

#include <functional>
#include <vector>
#include <set>
//#include "parameters.h"
#include "utils.h"

struct ModelParams;

// if the target needs to be scaled (into [-1,1]) before training, we store
// everything in this struct, that is required to invert the scaling after training
struct Scaler
{
    double data_min, data_max;
    double feature_min, feature_max;
    double scale, min_;
    bool scaling_required;
    Scaler(){};
    Scaler(double min_val, double max_val, double fmin, double fmax, bool scaling_required);
};

// basic wrapper around our data:
//  - matrix X
//  - target y
//  - vector for the samples' gradients (which get constantly updated)
//  - some useful attributes
struct DataSet
{
    // constructors
    DataSet();
    DataSet(VVD X, std::vector<double> y);

    // fields
    VVD X;
    std::vector<double> y;
    std::vector<double> gradients;
    int length, num_x_cols;
    bool empty;
    Scaler scaler;
    std::string name;

    // methods
    void add_row(std::vector<double> xrow, double yval);
    void scale_y(ModelParams &params, double lower, double upper);
    void scale_X_columns(ModelParams &params);
    void shuffle_dataset();
    void shuffle_dataset(std::mt19937 rng);
    DataSet get_subset(std::vector<int> &indices);
    DataSet remove_rows(std::vector<int> &indices);
    /**
     * @brief Get the first n elements of the dataset (i.e. the first n feature
     * vectors, the first n target values and the first n gradients).
     *
     * @param n how many elements to take
     * @return DataSet the first n elements in a new dataset (has no side effect
     * on the original dataset).
     */
    DataSet take(int n);
    /**
     * @brief Drop the first n elements of the dataset (i.e. keep only the
     * remaining elements).
     *
     * @param n how many elements to drop
     * @return DataSet the remaining elements in a new dataset (has no side
     * effect on the original dataset).
     */
    DataSet drop(int n);
    /**
     * @brief Split the dataset into two new datasets, where the first (left)
     * dataset contains all elements whose gradients satisfy the predicate `p`
     * and the second (right) dataset contains all the elements whose gradients
     * no not satisfy the predicate `p`.
     *
     * @param p the predicate applied to each samples gradient
     * @return std::pair<DataSet, DataSet> left: all satisfying samples in a new
     * dataset; right: all dissatisfying samples in a new dataset (no side
     * effect on the original dataset).
     */
    std::pair<DataSet, DataSet> partition_by_gradients(std::function<bool(double)> p);

    /**
     * @param bound which value to clip the gradients to
     * @return DataSet a new dataset whose gradients are clipped to bound.
     */
    DataSet clipped_gradients(double bound);
};

// wrapper around 2 DataSets that belong together
struct TrainTestSplit
{
    DataSet train;
    DataSet test;
    TrainTestSplit(DataSet train, DataSet test) : train(train), test(test){};
    TrainTestSplit(){};
};

// method declarations
void inverse_scale_y(ModelParams &params, Scaler &scaler, std::vector<double> &vec);
TrainTestSplit train_test_split_random(DataSet &dataset, double train_ratio = 0.70, bool shuffle = false);
std::vector<TrainTestSplit *> create_cross_validation_inputs(DataSet *dataset, int folds);
/**
 * @brief Join two datasets (put samples of ds2 behind the samples of ds1).
 *
 * @return DataSet a new joined dataset (no side effects on ds1 and ds2).
 * The attribute num_x_cols is the maximum of ds1.num_x_cols and ds2.num_x_cols.
 * The attribute name is just ds1.name (ds2's name is ignored).
 * The attribute scaler is just ds2.scaler (ds2's scaler is ignored). The caller
 * has the responsibility to make sure, that ds1's scaler is compatible with
 * the samples of ds2.
 */
DataSet join(const DataSet &ds1, const DataSet &ds2);

#endif /* DATA_H */

#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include <random>
#include <vector>
#include <fstream>
#include "custom_cauchy.h"
#include "dp_tree.h"
#include "parameters.h"
#include "data.h"

class DPEnsemble
{
public:
    // constructors
    DPEnsemble(ModelParams *params);
    ~DPEnsemble();

    // fields
    std::vector<DPTree> trees;

    // methods
    void train(DataSet *dataset);
    std::vector<double> predict(VVD &X);

private:
    // fields
    ModelParams *params;
    DataSet *dataset;
    double init_score;
    std::vector<double> grid;
    std::mt19937 rng;

    // methods
    void update_gradients(std::vector<double> &gradients, int trial_index);
};

#endif // DPENSEMBLE_H
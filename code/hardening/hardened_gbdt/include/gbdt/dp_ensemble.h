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
    DPEnsemble(ModelParams *params, std::mt19937_64 rng);
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
    custom_cauchy::CustomCauchy *noise_distribution;

    // methods
    void update_gradients(std::vector<double> &gradients, int tree_index);
};

#endif // DPENSEMBLE_H
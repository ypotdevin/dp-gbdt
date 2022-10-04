#ifndef DPENSEMBLE_H
#define DPENSEMBLE_H

#include <random>
#include <vector>
#include <fstream>
#include "custom_cauchy.h"
#include "dp_tree.h"

struct DataSet;
struct ModelParams;

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
    /**
     * @brief Training as Li et al. described it originally (differentially private,
     * but no tree rejection / ensemble optimization).
     *
     * @return vector<DPTree> a readily trained ensemble of DP trees
     */
    void vanilla_training_loop();
    //vector<DPTree> leaky_opt_training_loop(ModelParams &mp, DataSet &ds);

    /**
     * @brief Training mostly as Li et al. described it originally, but performing
     * tree rejection / ensemble optimization - i.e. reject trees which do not
     * decrease the ensemble loss.
     *
     * @return vector<DPTree> a readily trained ensemble of DP trees
     */
    void tree_rejection_training_loop();
};
#endif // DPENSEMBLE_H
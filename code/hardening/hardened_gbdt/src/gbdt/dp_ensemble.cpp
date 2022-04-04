#include <numeric>
#include <algorithm>
#include <iostream>
#include "dp_ensemble.h"
#include "constant_time.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "utils.h"
#include "loss.h"

extern std::ofstream verification_logfile;
extern size_t cv_fold_index;
extern bool VERIFICATION_MODE;

using namespace std;

/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    if (parameters->privacy_budget <= 0)
    {
        throw std::runtime_error("hardened gbdt cannot be run with pb<=0");
    }

    // prepare the linspace grid
    if (is_true(params->use_grid))
    {
        double grid_range = std::get<1>(params->grid_borders) - std::get<0>(params->grid_borders);
        double step_size = params->grid_step_size;
        int grid_size = (int)grid_range / step_size;
        this->grid = std::vector<double>(grid_size, 0);
        int counter = 0;
        std::generate(this->grid.begin(), this->grid.end(), [&counter, &step_size]() mutable
                      { return counter++ * step_size; });
    }
}

DPEnsemble::~DPEnsemble()
{
    for (auto tree : trees)
    {
        tree.delete_tree(tree.root_node);
    }
}

/** Methods */
// Note: The training has side effects on the dataset (affecting especially
// the gradients and the dataset rows).
void DPEnsemble::train(DataSet *dataset)
{
    this->dataset = dataset;
    int original_length = dataset->length;

    // compute initial prediction
    this->init_score = params->task->compute_init_score(dataset->y);
    LOG_DEBUG("Training initialized with score: {1}", init_score);

    auto prev_loss = std::numeric_limits<double>::infinity();

    // each tree gets the full pb, as they train on distinct data
    TreeParams tree_params;
    tree_params.tree_privacy_budget = params->privacy_budget;

    // train all trees
    for (int tree_index = 0; tree_index < params->nb_trees; tree_index++)
    {

        if (VERIFICATION_MODE)
        {
            VERIFICATION_LOG("Tree {0} CV-Ensemble {1}", tree_index, cv_fold_index);
        }

        // update/init gradients
        update_gradients(dataset->gradients, tree_index);

        // sensitivity for internal nodes
        tree_params.delta_g = 3 * pow(params->l2_threshold, 2);

        // sensitivity for leaves
        if (is_true(params->gradient_filtering) && !is_true(params->leaf_clipping))
        {
            // you can only "turn off" leaf clipping if GDF is enabled!
            tree_params.delta_v = params->l2_threshold / (1 + params->l2_lambda);
        }
        else
        {
            tree_params.delta_v = std::min((double)(params->l2_threshold / (1 + params->l2_lambda)),
                                           2 * params->l2_threshold * pow(1 - params->learning_rate, tree_index));
        }

        // determine number of rows
        int number_of_rows = 0;
        if (is_true(params->balance_partition))
        {
            // num_unused_rows / num_remaining_trees
            number_of_rows = dataset->length / (params->nb_trees - tree_index);
        }
        else
        {
            // line 8 of Algorithm 2 from the paper
            number_of_rows = (original_length * params->learning_rate *
                              std::pow(1 - params->learning_rate, tree_index)) /
                             (1 - std::pow(1 - params->learning_rate, params->nb_trees));
            if (number_of_rows == 0)
            {
                throw std::runtime_error("Warning: tree is not getting any samples");
            }
        }

        // this vector indicates which sample rows will be used for the next tree
        vector<int> tree_indices(dataset->length);

        // gradient-based data filtering
        if (is_true(params->gradient_filtering))
        {

            // divide samples into rejected/remaining gradients
            std::vector<int> reject_indices(dataset->length, 0), remaining_indices(dataset->length, 0);
            for (int i = 0; i < dataset->length; i++)
            {
                double curr_grad = dataset->gradients[i];
                bool reject = constant_time::logical_or(curr_grad<-params->l2_threshold, curr_grad> params->l2_threshold);
                reject_indices[i] = reject;
                remaining_indices[i] = constant_time::logical_not(reject);
            }

            int remaining_count = std::accumulate(remaining_indices.begin(), remaining_indices.end(), 0);
            LOG_INFO("GDF: {1} of {2} rows fulfill gradient criterion",
                     remaining_count, dataset->length);

            /** first use as many "remaining" samples as possible */

            // generate random index permutation
            std::vector<int> permutation(dataset->length);
            std::iota(std::begin(permutation), std::end(permutation), 1); // [1,2,3,...]
            if (!VERIFICATION_MODE)
            {
                std::random_shuffle(permutation.begin(), permutation.end());
            }
            // zero out all elements that are not part of the remaining array
            for (auto &elem : permutation)
            {
                elem *= remaining_indices[elem - 1];
            }
            std::transform(permutation.begin(), permutation.end(), permutation.begin(), [](int &c)
                           { return c - 1; }); // make it 0-indexed

            // put corresponding rows into tree_indices
            int taken_rows = 0;
            for (int i = 0; i < dataset->length; i++)
            {
                bool use_row = constant_time::logical_and(permutation[i] != -1, taken_rows < number_of_rows);
                taken_rows += (int)use_row;
                // touch the entire tree_samples vector, to hide which one is added
                for (int j = 0; j < dataset->length; j++)
                {
                    tree_indices[j] = constant_time::select(constant_time::logical_and(j == permutation[i], use_row), 1, tree_indices[j]);
                }
            }

            /** if necessary, fill up with (randomly chosen and clipped) rejected samples */

            int num_additional_samples_required = constant_time::max(number_of_rows - remaining_count, 0);
            LOG_INFO("GDF: filling up with {1} rows (clipping those gradients)", num_additional_samples_required);

            // generate random index permutation
            std::iota(std::begin(permutation), std::end(permutation), 1); // [1,2,3,...]
            if (!VERIFICATION_MODE)
            {
                std::random_shuffle(permutation.begin(), permutation.end());
            }
            // zero out all elements that are not part of the rejected array
            for (auto &elem : permutation)
            {
                elem *= reject_indices[elem - 1];
            }
            std::transform(permutation.begin(), permutation.end(), permutation.begin(), [](int &c)
                           { return c - 1; }); // make it 0-indexed

            // put corresponding rows into tree_indices
            taken_rows = 0;
            for (int i = 0; i < dataset->length; i++)
            {
                // use row iff the row is part of "rejected_indices" and we still need more samples
                bool use_row = constant_time::logical_and(permutation[i] != -1, taken_rows < num_additional_samples_required);
                taken_rows += (int)use_row;
                // clip gradient if this row is used
                double clipped_gradient = clamp(dataset->gradients[i], -params->l2_threshold, params->l2_threshold);
                dataset->gradients[i] = constant_time::select(use_row, clipped_gradient, dataset->gradients[i]);
                // touch the entire tree_indices vector, to hide which one is added
                for (int j = 0; j < dataset->length; j++)
                {
                    // if we are at the right element, write 1, otherwise keep content
                    tree_indices[j] = constant_time::select(constant_time::logical_and(j == permutation[i], use_row), 1, tree_indices[j]);
                }
            }
        }
        else
        {
            // no GDF, just randomly select <number_of_rows> rows. This should not require hardening.
            // Note, this causes the leaves to be clipped after building the tree.
            vector<int> all_indices(dataset->length);
            std::iota(std::begin(all_indices), std::end(all_indices), 0);
            if (!VERIFICATION_MODE)
            {
                std::random_shuffle(all_indices.begin(), all_indices.end());
            }
            for (int i = 0; i < number_of_rows; i++)
            {
                tree_indices[all_indices[i]] = 1;
            }
        }

        DataSet tree_dataset = dataset->get_subset(tree_indices);

        LOG_DEBUG(YELLOW("Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                  tree_index, tree_params.tree_privacy_budget, tree_dataset.length);

        // build tree
        LOG_INFO("Building dp-tree-{1} using {2} samples...", tree_index, tree_dataset.length);
        DPTree tree = DPTree(params, &tree_params, &tree_dataset, tree_index, this->grid);
        tree.fit();
        trees.push_back(tree);

        /* We insert the tree-rejection here since it is easier to manage
         * the consumed / not consumed training sample here, instead of
         * after the case distinction on <params->use_dp>.
         * Note that this implies the optimization is only available if
         * using differential privacy -- not when DP is turned off (keep
         * that in mind when comparing plots of these settings.
         */
        bool keep_new_tree;
        if (!params->optimize)
        {
            // no optimization -> keep each suggested tree
            keep_new_tree = true;
        }
        else
        {
            // optimization -> keeping a tree depends on its performance gain
            auto raw_predictions = predict(dataset->X);
            std::vector<double> differences = std::vector<double>(dataset->length);
            std::transform(raw_predictions.begin(), raw_predictions.end(), dataset->y.begin(), differences.begin(), std::minus<double>());
            auto mean = compute_mean(differences);
            auto stdev = compute_stdev(differences, mean);
            LOG_DEBUG("#error_evolution# --- mean={1}; stdev={2}", mean, stdev);

            double current_loss;
            if (params->leaky_opt && std::isnan(params->optimization_privacy_budget))
            {
                current_loss = compute_rmse(differences);
                keep_new_tree = current_loss >= 0.0 && current_loss < prev_loss;
            }
            else if (!params->leaky_opt && !std::isnan(params->optimization_privacy_budget))
            {
                current_loss = dp_rms_cauchy(differences, params->optimization_privacy_budget, params->error_upper_bound);
                keep_new_tree = current_loss < prev_loss;
            }
            else
            {
                throw std::runtime_error("illegal combination of of optimization, leaky optimization and DP optimization budget");
            }
            if (keep_new_tree)
            {
                prev_loss = current_loss;
            }
            LOG_INFO("#loss_evolution# --- fitting decision tree {1}; previous loss: {2}; current loss: {3}", tree_index, prev_loss, current_loss);
        }

        if (keep_new_tree)
        {
            *dataset = dataset->remove_rows(tree_indices);
        }
        else
        {
            trees.pop_back();
        }
        LOG_DEBUG(
            "#tree_evolution# --- ensemble {1} decision tree {2}. Number of trees in ensemble: {3}. Instances left: {4}.",
            keep_new_tree ? "includes" : "excludes",
            tree_index,
            trees.size(),
            dataset->length
        );
    }
}

// Predict values from the ensemble of gradient boosted trees
vector<double> DPEnsemble::predict(VVD &X)
{
    vector<double> predictions(X.size(), 0);
    for (auto tree : trees)
    {
        vector<double> pred = tree.predict(X);

        std::transform(pred.begin(), pred.end(),
                       predictions.begin(), predictions.begin(), std::plus<double>());
    }

    double innit_score = this->init_score;
    double learning_rate = params->learning_rate;
    std::transform(predictions.begin(), predictions.end(), predictions.begin(),
                   [learning_rate, innit_score](double &c)
                   { return c * learning_rate + innit_score; });

    return predictions;
}

void DPEnsemble::update_gradients(vector<double> &gradients, int tree_index)
{
    if (tree_index == 0)
    {
        // init gradients
        vector<double> init_scores(dataset->length, init_score);
        gradients = params->task->compute_gradients(dataset->y, init_scores);
    }
    else
    {
        // update gradients
        vector<double> y_pred = predict(dataset->X);
        gradients = (params->task)->compute_gradients(dataset->y, y_pred);
    }
    if (VERIFICATION_MODE)
    {
        double sum = std::accumulate(gradients.begin(), gradients.end(), 0.0);
        sum = sum < 0 && sum >= -1e-10 ? 0 : sum; // avoid "-0.00000.. != 0.00000.."
        VERIFICATION_LOG("GRADIENTSUM {0:.8f}", sum);
    }
}

#include <numeric>
#include <algorithm>
#include <iostream>
#include "dp_ensemble.h"
#include "constant_time.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "tree_rejection.h"
#include "utils.h"
#include "loss.h"

extern size_t cv_fold_index;

using namespace std;

/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    if (parameters->privacy_budget <= 0)
    {
        throw std::runtime_error("hardened gbdt cannot be run with pb<=0");
    }

    this->rng = rng;

    // prepare the linspace grid
    if (params->use_grid)
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

    // each tree gets the full pb, as they train on distinct data
    TreeParams tree_params;
    double ensemble_budget = params->ensemble_rejector_budget_split * params->privacy_budget;
    double rejector_budget = (1.0 - params->ensemble_rejector_budget_split) * params->privacy_budget;
    tree_params.tree_privacy_budget = ensemble_budget;
    this->params->tree_rejector->set_total_privacy_budget(rejector_budget);

    int tree_index = 0;
    // train all trees
    for (int trial_index = 0; trial_index < params->n_trials; trial_index++)
    {
        if (tree_index >= params->n_trees_to_accept)
        {
            // If there are already as many trees accepted as allowed, early stop
            break;
        }
        // update/init gradients
        update_gradients(dataset->gradients, trial_index);

        // sensitivity for internal nodes
        tree_params.delta_g = 3 * pow(params->l2_threshold, 2);

        // sensitivity for leaves
        if (params->gradient_filtering && !params->leaf_clipping)
        {
            // you can only "turn off" leaf clipping if GDF is enabled!
            tree_params.delta_v = params->l2_threshold / (1 + params->l2_lambda);
        }
        else
        {
            tree_params.delta_v = std::min((double)(params->l2_threshold / (1 + params->l2_lambda)),
                                           2 * params->l2_threshold * pow(1 - params->learning_rate, trial_index));
        }

        // determine number of rows
        int number_of_rows = 0;
        if (params->balance_partition)
        {
            int num_remaining_trees = std::min(
                params->n_trees_to_accept - tree_index,
                params->n_trials - trial_index
            );
            // num_unused_rows / num_remaining_trees
            number_of_rows = dataset->length / num_remaining_trees;
        }
        else
        {
            // line 8 of Algorithm 2 from the paper
            number_of_rows = (original_length * params->learning_rate *
                              std::pow(1 - params->learning_rate, tree_index)) /
                             (1 - std::pow(1 - params->learning_rate, params->n_trees_to_accept));
            if (number_of_rows == 0)
            {
                throw std::runtime_error("Warning: tree is not getting any samples");
            }
        }

        // this vector indicates which sample rows will be used for the next tree
        vector<int> tree_indices(dataset->length);

        // gradient-based data filtering
        if (params->gradient_filtering)
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
            std::random_shuffle(permutation.begin(), permutation.end());
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
            std::random_shuffle(permutation.begin(), permutation.end());
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
            std::random_shuffle(all_indices.begin(), all_indices.end());
            for (int i = 0; i < number_of_rows; i++)
            {
                tree_indices[all_indices[i]] = 1;
            }
        }

        DataSet tree_dataset = dataset->get_subset(tree_indices);

        LOG_DEBUG(YELLOW("Trial-Tree {1:2d}: receives pb {2:.2f} and will train on {3} instances"),
                  trial_index, tree_params.tree_privacy_budget, tree_dataset.length);

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
        auto raw_predictions = predict(dataset->X);
        auto reject_new_tree = params->tree_rejector->reject_tree(dataset->y, raw_predictions);

        if (reject_new_tree)
        {
            trees.pop_back();
        }
        else
        {
            *dataset = dataset->remove_rows(tree_indices);
            tree_index += 1;
        }
        LOG_INFO(
            "### diagnosis value 07 ### - ensemble {1} trial-tree {2}. Number of trees in ensemble: {3}. Instances left: {4}.",
            reject_new_tree ? "excludes" : "includes",
            trial_index,
            trees.size(),
            dataset->length);
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

void DPEnsemble::update_gradients(vector<double> &gradients, int trial_index)
{
    if (trial_index == 0)
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
}

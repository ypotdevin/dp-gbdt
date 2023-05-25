#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "gbdt/dp_tree.h"
#include "gbdt/laplace.h"
#include "logging.h"
#include "constant_time.h"
#include "spdlog/spdlog.h"

using namespace std;

/** Constructors */

DPTree::DPTree(
    ModelParams *params,
    TreeParams *tree_params,
    DataSet *dataset,
    size_t tree_index,
    const std::vector<std::vector<double>> &grid) : params(params),
                                                    tree_params(tree_params),
                                                    dataset(dataset),
                                                    tree_index(tree_index),
                                                    grid(grid)

{
    this->random_unit_doubles = std::uniform_real_distribution<double>(0.0, 1.0);
    // only need to transpose X once
    X_transposed = VVD(dataset->num_x_cols, vector<double>(dataset->length));
    for (int row = 0; row < dataset->length; row++)
    {
        for (int col = 0; col < dataset->num_x_cols; col++)
        {
            X_transposed[col][row] = (dataset->X)[row][col];
        }
    }
}

/** Methods */

// Fit the tree to the data
void DPTree::fit()
{
    // keep track which samples will be available in a node for spliting (1=live)
    vector<int> live_samples(dataset->length, 1);

    this->root_node = make_tree_dfs(0, live_samples);

    // leaf clipping. Note, it can only be disabled if GDF is enabled.
    if (this->params->leaf_clipping or !this->params->gradient_filtering)
    {
        double threshold = this->params->l2_threshold *
                           std::pow((1 - this->params->learning_rate), tree_index);
        for (auto &node : this->leaves)
        {
            node->prediction = clamp(node->prediction, -threshold, threshold);
        }
    }

    // add laplace noise to leaf values
    double privacy_budget_for_leaf_nodes = tree_params->tree_privacy_budget / 2;
    double laplace_scale = tree_params->delta_v / privacy_budget_for_leaf_nodes;
    add_laplacian_noise(laplace_scale);
}

// Recursively build tree, DFS approach, first instance returns root node
shared_ptr<TreeNode> DPTree::make_tree_dfs(int current_depth, vector<int> live_samples)
{
    int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
    // sgx_printf("live_size: %i\n", live_size);
    bool reached_max_depth = (current_depth == this->params->max_depth);
    bool not_enough_live_samples = (live_size < this->params->min_samples_split);
    bool create_leaf_node = constant_time::logical_or(reached_max_depth,
                                                      not_enough_live_samples);
    if (create_leaf_node)
    {
        return make_leaf_node(current_depth, live_samples);
    }

    // find best split
    shared_ptr<TreeNode> node = find_best_split(X_transposed,
                                                dataset->gradients,
                                                live_samples,
                                                current_depth,
                                                create_leaf_node);

    bool categorical = std::find(this->params->cat_idx.begin(),
                                 this->params->cat_idx.end(),
                                 node->split_attr) !=
                       this->params->cat_idx.end();

    // prepare the new R/L live_samples to continue the recursion, constant time
    vector<int> lhs(live_samples.size(), 0);
    vector<int> rhs(live_samples.size(), 0);

    samples_left_right_partition(lhs,
                                 rhs,
                                 X_transposed,
                                 live_samples,
                                 node->split_attr,
                                 node->split_value,
                                 categorical);

    vector<int> left_live_samples(live_samples.size(), 0);
    vector<int> right_live_samples(live_samples.size(), 0);
    for (size_t i = 0; i < live_samples.size(); i++)
    {
        left_live_samples[i] = lhs[i] * (live_samples[i] == 1);
        right_live_samples[i] = rhs[i] * (live_samples[i] == 1);
    }

    node->left = make_tree_dfs(current_depth + 1, left_live_samples);
    node->right = make_tree_dfs(current_depth + 1, right_live_samples);

    return node;
}

shared_ptr<TreeNode> DPTree::make_leaf_node(int current_depth, vector<int> &live_samples)
{
    shared_ptr<TreeNode> leaf = shared_ptr<TreeNode>(new TreeNode(true));
    leaf->depth = current_depth;

    // compute prediction
    double gradients_sum = 0;
    int live_size = 0;
    for (size_t index = 0; index < live_samples.size(); index++)
    {
        gradients_sum += dataset->gradients[index] * live_samples[index];
        live_size += live_samples[index];
    }
    leaf->prediction = (-1 * gradients_sum /
                        (live_size + this->params->l2_lambda));
    leaves.push_back(leaf);

    return leaf;
}

vector<double> DPTree::predict(VVD &X)
{
    vector<double> predictions;
    predictions.reserve(X.size());
    // iterate over all samples
    for (auto &row : X)
    {
        double pred = _predict(&row, this->root_node);
        predictions.push_back(pred);
    }

    return predictions;
}

// recursively walk through decision tree
double DPTree::_predict(vector<double> *row, shared_ptr<TreeNode> node)
{
    if (node->is_leaf)
    {
        return node->prediction;
    }

    bool categorical = std::find(this->params->cat_idx.begin(),
                                 this->params->cat_idx.end(),
                                 node->split_attr) !=
                       this->params->cat_idx.end();

    double next_level_prediction = 0.0;

    if (node->depth < this->params->max_depth)
    {
        double row_val = (*row)[node->split_attr];

        double left_result = _predict(row, node->left);
        double right_result = _predict(row, node->right);
        // to hide the real path a sample row takes, we will go down both paths at every
        // internal node.
        // Further we hide whether the current node splits on a categorical/numerical feature.
        // Which is kinda unnecessary, as the proof gives this
        // to the adversary. however it might allow for a tighter proof later.
        next_level_prediction = constant_time::select(
            categorical,
            constant_time::select((row_val == node->split_value),
                                  left_result,
                                  right_result),
            constant_time::select((row_val < node->split_value),
                                  left_result,
                                  right_result));
    }

    // decide whether to take the current node's prediction, or the prediction of its successors
    return constant_time::select(node->is_leaf,
                                 node->prediction,
                                 next_level_prediction);
}

// find best split of data using the exponential mechanism
shared_ptr<TreeNode> DPTree::find_best_split(
    VVD &X_transposed,
    vector<double> &gradients_live,
    vector<int> &live_samples,
    int current_depth,
    bool create_leaf_node)
{
    double privacy_budget_for_node;
    if (this->params->use_decay)
    {
        if (current_depth == 0)
        {
            privacy_budget_for_node =
                tree_params->tree_privacy_budget /
                (2 * pow(2, this->params->max_depth + 1) + 2 * pow(2, current_depth + 1));
        }
        else
        {
            privacy_budget_for_node = tree_params->tree_privacy_budget /
                                      (2 * pow(2, current_depth + 1));
        }
    }
    else
    {
        privacy_budget_for_node = (tree_params->tree_privacy_budget) /
                                  (2 * this->params->max_depth);
    }

    vector<SplitCandidate> probabilities;
    int lhs_size;

    // iterate over features
    for (int feature_index = 0;
         feature_index < dataset->num_x_cols;
         feature_index++)
    {

        bool categorical = std::find(
                               this->params->cat_idx.begin(),
                               this->params->cat_idx.end(),
                               feature_index) !=
                           this->params->cat_idx.end();

        /*
            TODO: At some time decide which variant should be final.
            For now, it doesn't seem to be reasonable to make this a
            hyperparameter (again).
        */
        bool use_grid = true;
        if (use_grid)
        {
            // std::vector<double> feature_values;
            // if (categorical)
            // {
            //     feature_values = this->params->cat_values[feature_index];
            // }
            // else
            // {
            //     feature_values = this->grid.at(feature_index);
            // }
            for (double feature_value : this->grid.at(feature_index))
            {
                // compute gain
                double gain = compute_gain(X_transposed,
                                           gradients_live,
                                           live_samples,
                                           feature_index,
                                           feature_value,
                                           lhs_size,
                                           categorical);
                bool no_gain = (gain == -1.);
                // if gain is -1 (aka the split guides all samples to the same
                // child -> useless split) then the gain of this split is set
                // to 0.
                gain = constant_time::select(
                    no_gain,
                    0.0,
                    gain);

                // Gi = epsilon_nleaf * Gi / (2 * delta_G)
                gain = (privacy_budget_for_node * gain) /
                       (2 * tree_params->delta_g);

                SplitCandidate candidate = SplitCandidate(feature_index,
                                                          feature_value,
                                                          gain);
                candidate.lhs_size = lhs_size;
                candidate.rhs_size = std::accumulate(
                                         live_samples.begin(),
                                         live_samples.end(),
                                         0) -
                                     lhs_size;
                probabilities.push_back(candidate);
            }
        }
        else
        {
            // auto feature_values = X_transposed[feature_index];
            /* Don't use the grid - use the feature values to split */
            for (size_t row = 0; row < live_samples.size(); row++)
            {
                double feature_value = X_transposed[feature_index][row];

                // compute gain
                double gain = compute_gain(X_transposed,
                                           gradients_live,
                                           live_samples,
                                           feature_index,
                                           feature_value,
                                           lhs_size,
                                           categorical);

                bool row_not_live = constant_time::logical_not(live_samples[row]);
                bool no_gain = (gain == -1.);
                // if either the row is not live, or the gain is -1 (aka the
                // split guides all samples to the same child -> useless split)
                // then the gain of this split is set to 0.
                gain = constant_time::select(
                    constant_time::logical_or(no_gain, row_not_live),
                    0.0,
                    gain);

                // Gi = epsilon_nleaf * Gi / (2 * delta_G)
                gain = (privacy_budget_for_node * gain) /
                       (2 * tree_params->delta_g);

                SplitCandidate candidate = SplitCandidate(feature_index,
                                                          feature_value,
                                                          gain);
                candidate.lhs_size = lhs_size;
                candidate.rhs_size = std::accumulate(
                                         live_samples.begin(),
                                         live_samples.end(),
                                         0) -
                                     lhs_size;
                probabilities.push_back(candidate);
            }
        }
    }

    // choose a split using the exponential mechanism
    int index = exponential_mechanism(probabilities);

    // start by constructing a leaf node
    shared_ptr<TreeNode> node = make_leaf_node(current_depth, live_samples);

    // if an internal node should be created, change attributes accordingly
    bool create_internal_node = constant_time::logical_and(
        index != -1,
        constant_time::logical_not(create_leaf_node));

    // go through all candidates, to hide which one was chosen
    SplitCandidate chosen_one = SplitCandidate(0, 0, 0);
    chosen_one.lhs_size = 0;
    chosen_one.rhs_size = 0;
    for (int i = 0; i < (int)probabilities.size(); i++)
    {
        bool cond = (i == index);
        chosen_one.feature_index = constant_time::select(
            cond,
            probabilities[i].feature_index,
            chosen_one.feature_index);
        chosen_one.split_value = constant_time::select(
            cond,
            probabilities[i].split_value,
            chosen_one.split_value);
        chosen_one.gain = constant_time::select(
            cond,
            probabilities[i].gain,
            chosen_one.gain);
        chosen_one.lhs_size = constant_time::select(
            cond,
            probabilities[i].lhs_size,
            chosen_one.lhs_size);
        chosen_one.rhs_size = constant_time::select(
            cond,
            probabilities[i].rhs_size,
            chosen_one.rhs_size);
    }
    node->split_attr = constant_time::select(create_internal_node,
                                             chosen_one.feature_index,
                                             node->split_attr);
    node->split_value = constant_time::select(create_internal_node,
                                              chosen_one.split_value,
                                              node->split_value);
    node->split_gain = constant_time::select(create_internal_node,
                                             chosen_one.gain,
                                             node->split_gain);
    node->lhs_size = constant_time::select(create_internal_node,
                                           chosen_one.lhs_size,
                                           node->lhs_size);
    node->rhs_size = constant_time::select(create_internal_node,
                                           chosen_one.rhs_size,
                                           node->rhs_size);
    node->is_leaf = constant_time::logical_not(create_internal_node);

    if (create_leaf_node)
    {
        int live_size = std::accumulate(live_samples.begin(), live_samples.end(), 0);
        LOG_DEBUG("max_depth ({1}) or min_samples ({2})-> leaf (pred={3:.2f})",
                  current_depth, live_size, node->prediction);
    }
    LOG_DEBUG("split value: {1}", node->split_value);
    return node;
}

/*
    Computes the gain of a split

               sum(elem : IL)^2  +  sum(elem : IR)^2
    G(IL,IR) = ----------------     ----------------
                |IL| + lambda        |IR| + lambda
*/
double DPTree::compute_gain(
    VVD &X_transposed,
    vector<double> &gradients_live,
    vector<int> &live_samples,
    int feature_index,
    double feature_value,
    int &lhs_size,
    bool categorical)
{
    // partition sample rows into lhs / rhs
    vector<int> lhs(live_samples.size(), 0);
    vector<int> rhs(live_samples.size(), 0);

    samples_left_right_partition(lhs,
                                 rhs,
                                 X_transposed,
                                 live_samples,
                                 feature_index,
                                 feature_value,
                                 categorical);

    int _lhs_size = std::accumulate(lhs.begin(), lhs.end(), 0);
    int _rhs_size = std::accumulate(rhs.begin(), rhs.end(), 0);

    lhs_size = _lhs_size;

    // if all samples go on the same side it's useless to split on this value
    bool useless_split = constant_time::logical_or(_lhs_size == 0, _rhs_size == 0);

    double lhs_gain = 0, rhs_gain = 0;
    for (size_t index = 0; index < live_samples.size(); index++)
    {
        lhs_gain += lhs[index] * (gradients_live)[index];
        rhs_gain += rhs[index] * (gradients_live)[index];
    }
    lhs_gain = std::pow(lhs_gain, 2) / (_lhs_size + this->params->l2_lambda);
    rhs_gain = std::pow(rhs_gain, 2) / (_rhs_size + this->params->l2_lambda);

    double total_gain = lhs_gain + rhs_gain;
    // total_gain = max(total_gain, 0.0);
    total_gain = constant_time::select(total_gain < 0.0, 0.0, total_gain);

    // useless split -> return (-1) instead
    total_gain = constant_time::select(useless_split, -1., total_gain);

    return total_gain;
}

void DPTree::samples_left_right_partition(
    vector<int> &lhs,
    vector<int> &rhs,
    VVD &samples,
    vector<int> &live_samples,
    int feature_index,
    double feature_value,
    bool categorical)
{
    for (size_t row = 0; row < live_samples.size(); row++)
    {
        if (categorical)
        {
            lhs[row] = constant_time::select(live_samples[row], (int)(samples[feature_index][row] == feature_value), 0);
            rhs[row] = constant_time::select(live_samples[row], (int)(samples[feature_index][row] != feature_value), 0);
        }
        else
        {
            lhs[row] = constant_time::select(live_samples[row], (int)(samples[feature_index][row] < feature_value), 0);
            rhs[row] = constant_time::select(live_samples[row], (int)(samples[feature_index][row] >= feature_value), 0);
        }
    }
}

// Computes probabilities from the gains. (Larger gain -> larger probability to
// be chosen for split). Then a cumulative distribution function is created from
// these probabilities. Then we can sample from it using a RNG.
// The function returns the index of the chosen split.
int DPTree::exponential_mechanism(vector<SplitCandidate> &candidates)
{
    // if no split has a positive gain -> return -1. Node will become a leaf
    int num_viable_candidates = 0;
    for (auto candidate : candidates)
    {
        num_viable_candidates += (candidate.gain > 0);
    }
    bool no_split_available = (num_viable_candidates == 0);

    // calculate the probabilities from the gains
    vector<double> gains, probabilities, partials(candidates.size());
    for (auto p : candidates)
    {
        gains.push_back(p.gain);
    }
    double lse = log_sum_exp(gains);
    for (auto prob : candidates)
    {
        // let probability be 0 when gain is <= 0
        double probability = constant_time::select(prob.gain <= 0, 0.0, exp(prob.gain - lse));
        probabilities.push_back(probability);
    }

    // create a cumulative distribution from the probabilities.
    // all values will be in [0,1]
    std::partial_sum(probabilities.begin(), probabilities.end(), partials.begin());

    // std::uniform_real_distribution<double> random_unit_doubles(0.0, 1.0);
    double rand01 = this->random_unit_doubles(this->params->rng);

    size_t result_index = 0;
    bool found = false;
    for (size_t index = 0; index < partials.size(); index++)
    {
        result_index = constant_time::select(found, result_index, index);
        found = (partials[index] >= rand01);
    }
    return constant_time::select(no_split_available, -1, (int)result_index);
}

void DPTree::
    add_laplacian_noise(double laplace_scale)
{
    LOG_DEBUG("Adding Laplace noise to leaves (Scale {1:.2f})", laplace_scale);

    Laplace lap(this->params->rng);

    // add noise from laplace distribution (to all nodes, for the hardened version)
    for (auto &node : leaves)
    {
        double noise = lap.return_a_random_variable(laplace_scale);
        node->prediction += noise;
        LOG_DEBUG("({1:.3f} -> {2:.8f})", node->prediction, node->prediction + noise);
    }
}
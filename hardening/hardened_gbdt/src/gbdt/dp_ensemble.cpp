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
#include "data.h"

namespace
{
    const double e = 2.71828182845904523536;
    TreeParams setup_tree_params(const ModelParams &mp,
                                 double tree_budget,
                                 int tree_index)
    {
        TreeParams tree_params;
        tree_params.tree_privacy_budget = tree_budget;
        tree_params.delta_g = 3 * pow(mp.l2_threshold, 2);
        if (mp.gradient_filtering && !mp.leaf_clipping)
        {
            // you can only "turn off" leaf clipping if GDF is enabled!
            tree_params.delta_v = mp.l2_threshold / (1 + mp.l2_lambda);
        }
        else
        {
            tree_params.delta_v = std::min(
                (double)(mp.l2_threshold / (1 + mp.l2_lambda)),
                2 * mp.l2_threshold * pow(1 - mp.learning_rate, tree_index));
        }
        return tree_params;
    }

    /**
     * @brief In case of unbalanced partitions, calculate the number of rows
     * for the current tree dataset as defined by line 8 of Li et al.'s
     * Algorithm 2.
     */
    int line_eight(
        int original_dataset_length,
        double learning_rate,
        int n_trees_to_accept,
        int tree_index)
    {
        int rows_per_step = (original_dataset_length * learning_rate *
                             std::pow(1.0 - learning_rate, tree_index + 1.0)) /
                            (1 - std::pow(1.0 - learning_rate, n_trees_to_accept));
        if (rows_per_step <= 0)
        {
            throw std::runtime_error("Warning: tree is not getting any samples");
        }
        return rows_per_step;
    }

    /**
     * @brief Extract up to num_rows rows from dataset, whose absolute gradient
     * value is at most l2_threshold. If there are not enough rows satisfying
     * this condition, fill up with the remaining rows, but clip their
     * gradients.
     *
     * @return std::pair<DataSet, DataSet> first component: those rows whose
     * absolute gradient naturally, or by clipping, is less than l2_threshold;
     * second component: the remaining rows
     */
    std::pair<DataSet, DataSet> gradient_filtered_data(
        DataSet &dataset,
        int num_rows,
        double l2_threshold)
    {
        DataSet tree_dataset;
        DataSet remaining_dataset;
        auto predicate = [l2_threshold](double gradient)
        { return std::abs(gradient) < l2_threshold; };
        auto ds_pair = dataset.partition_by_gradients(predicate);
        auto within_range = ds_pair.first;
        auto out_of_range = ds_pair.second;
        if (within_range.length >= num_rows)
        {
            // if enough rows satisfy the gradient condition
            tree_dataset = within_range.take(num_rows);
            auto excess_within_range = within_range.drop(num_rows);
            remaining_dataset = join(excess_within_range, out_of_range);
        }
        else
        {
            auto num_missing = num_rows - within_range.length;
            auto additional_within_range =
                out_of_range.clipped_gradients(l2_threshold).take(num_missing);
            tree_dataset = join(within_range, additional_within_range);
            remaining_dataset = out_of_range.drop(num_missing);
        }
        return std::pair<DataSet, DataSet>(tree_dataset, remaining_dataset);
    }

    /**
     * @return std::pair<DataSet, DataSet> first component: the rows dedicated
     * to the current tree; second component: the remaining rows
     */
    std::pair<DataSet, DataSet> setup_tree_data(
        DataSet &dataset,
        const ModelParams &mp,
        int n_steps,
        int original_dataset_length,
        int tree_index)
    {
        int rows_per_step;
        if (mp.balance_partition)
        {
            rows_per_step = dataset.length / (n_steps - tree_index);
        }
        else
        {
            rows_per_step = line_eight(original_dataset_length,
                                       mp.learning_rate,
                                       mp.n_trees_to_accept,
                                       tree_index);
        }

        if (mp.gradient_filtering)
        {
            return gradient_filtered_data(dataset,
                                          rows_per_step,
                                          mp.l2_threshold);
        }
        else
        {
            return std::pair<DataSet, DataSet>(dataset.take(rows_per_step),
                                               dataset.drop(rows_per_step));
        }
    }

    /**
     * @brief Generate the prediction of an ensemble including `tree`, based
     * upon the ensemble's previous prediction (without `tree`).
     */
    std::vector<double> incremental_predict(
        const std::vector<double> &old_prediction,
        double learning_rate,
        VVD &X,
        DPTree &tree)
    {
        auto tree_prediction = tree.predict(X);
        std::transform(
            old_prediction.begin(),
            old_prediction.end(),
            tree_prediction.begin(),
            tree_prediction.begin(),
            [learning_rate](double old_pred, double t_pred)
            { return old_pred + learning_rate * t_pred; });
        return tree_prediction;
    }
}

using namespace std;

/** Constructors */

DPEnsemble::DPEnsemble(ModelParams *parameters) : params(parameters)
{
    if (this->params->privacy_budget <= 0)
    {
        throw std::runtime_error("hardened gbdt cannot be run with pb<=0");
    }

    this->rng = rng;

    /* Setup data independent grid for tree node splitting later*/
    if (!(this->params->grid_lower_bounds.size() ==
              this->params->grid_upper_bounds.size() &&
          this->params->grid_upper_bounds.size() ==
              this->params->grid_step_sizes.size()))
    {
        throw std::runtime_error(
            "Number of grid bounds and step sizes do not match!");
    }
    else
    {
        for (size_t i = 0; i < this->params->grid_lower_bounds.size(); ++i)
        {
            this->_grid.push_back(
                numpy::linspace(
                    this->params->grid_lower_bounds.at(i),
                    this->params->grid_upper_bounds.at(i),
                    this->params->grid_step_sizes.at(i)));
        }
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
void DPEnsemble::train(DataSet &dataset)
{
    this->init_score = this->params->task->compute_init_score(dataset.y);
    this->init_gradients(dataset);
    LOG_DEBUG("Training initialized with ensemble-score: {1}", init_score);
    auto flavor = this->params->training_variant;
    if (flavor == "vanilla")
    {
        LOG_INFO("Running vanilla training (no optimization).");
        this->vanilla_training_loop(dataset);
    }
    else if (flavor == "dp_argmax_scoring")
    {
        LOG_INFO("Running optimized training.");
        this->dp_argmax_scoring_training_loop(dataset);
    }
    else
    {
        throw std::runtime_error("Training mode not recognized.");
    }
}

// Predict values from the ensemble of gradient boosted trees
vector<double> DPEnsemble::predict(VVD &X)
{
    vector<double> predictions(X.size(), 0);
    for (auto &tree : trees)
    {
        vector<double> pred = tree.predict(X);

        std::transform(pred.begin(),
                       pred.end(),
                       predictions.begin(),
                       predictions.begin(),
                       std::plus<double>());
    }

    double innit_score = this->init_score;
    double learning_rate = params->learning_rate;
    std::transform(predictions.begin(), predictions.end(), predictions.begin(),
                   [learning_rate, innit_score](double &c)
                   { return c * learning_rate + innit_score; });

    return predictions;
}

void DPEnsemble::vanilla_training_loop(DataSet &dataset)
{
    auto original_dataset_length = dataset.length;
    auto mp = *this->params;
    dataset.shuffle_dataset(mp.rng);
    auto n_steps = mp.n_trees_to_accept;
    auto step_budget = mp.privacy_budget; // parallel composition

    for (int tree_index = 0; tree_index < n_steps; ++tree_index)
    {
        LOG_INFO("tree_index={1}", tree_index);
        this->update_gradients(dataset);

        auto tree_params = setup_tree_params(mp, step_budget, tree_index);

        auto p = setup_tree_data(dataset,
                                 mp,
                                 n_steps,
                                 original_dataset_length,
                                 tree_index);
        auto tree_dataset = p.first;
        auto remaining_dataset = p.second;

        /* actual tree construction */
        DPTree tree = DPTree(this->params,
                             &tree_params,
                             &tree_dataset,
                             tree_index,
                             this->_grid);
        tree.fit();
        this->trees.push_back(tree);
        dataset = remaining_dataset;
    }
}

void DPEnsemble::dp_argmax_scoring_training_loop(DataSet &dataset)
{
    auto original_dataset_length = dataset.length;
    auto mp = *this->params;
    dataset.shuffle_dataset(mp.rng);

    auto n_steps = mp.n_trees_to_accept;
    auto step_budget = (mp.privacy_budget - mp.dp_argmax_privacy_budget) / 2.0; // parallel composition
    auto tree_budget = step_budget * mp.ensemble_rejector_budget_split;
    auto score_budget = (step_budget - tree_budget) /
                        static_cast<double>(mp.n_trees_to_accept); // division due to repetitive usage of dataset.X for tree scoring
    std::bernoulli_distribution biased_coin{mp.stopping_prob};

    int T = std::max((1.0 / mp.stopping_prob) *
                         std::log(2.0 / mp.dp_argmax_privacy_budget),
                     1.0 + 1.0 / (e * mp.stopping_prob));
    LOG_INFO("### diagnosis value 08 ### - T={1}", T);
    for (int tree_index = 0; tree_index < n_steps; ++tree_index)
    {
        this->update_gradients(dataset);
        auto tree_params = setup_tree_params(mp, tree_budget, tree_index);
        auto p = setup_tree_data(dataset,
                                 mp,
                                 n_steps,
                                 original_dataset_length,
                                 tree_index);
        auto tree_dataset = p.first;
        auto remaining_dataset = p.second;
        dp_argmax(dataset,
                  mp,
                  tree_params,
                  tree_dataset,
                  score_budget,
                  biased_coin,
                  T,
                  tree_index);
        dataset = remaining_dataset;
    }
    LOG_INFO(
        "### diagnosis value 10 ### - Finished training with n_accepted_trees={1}",
        this->trees.size());
}

void DPEnsemble::dp_argmax(
    DataSet &dataset,
    ModelParams &mp,
    TreeParams &tree_params,
    DataSet &tree_dataset,
    double score_budget,
    std::bernoulli_distribution &biased_coin,
    int T,
    int tree_index)
{
    auto ensemble_prediction = this->predict(dataset.X);
    auto ensemble_score = compute_rmse(ensemble_prediction, dataset.y);
    /* the actual generalized DP argmax algorithm from Liu and Talwar 2018 */
    for (int trial = 0; trial < T; ++trial)
    {
        LOG_DEBUG("### diagnosis value 09 ### - trial={1}", trial);
        DPTree tree = DPTree(this->params,
                             &tree_params,
                             &tree_dataset,
                             tree_index,
                             this->_grid);
        tree.fit();
        auto prediction_including_tree = incremental_predict(
            ensemble_prediction,
            mp.learning_rate,
            dataset.X,
            tree);
        auto score_including_tree = mp.tree_scorer->score_tree(
            score_budget,
            dataset.y,
            prediction_including_tree);
        LOG_DEBUG("score excluding new tree: {1}, score including new tree: {2}", ensemble_score, score_including_tree);
        LOG_INFO("### diagnosis value 02 ### - current rmse_approx={1}", score_including_tree);
        if (score_including_tree < ensemble_score)
        {
            LOG_INFO("generalized_dp_argmax: successful exit");
            this->trees.push_back(tree);
            return;
        }
        if (biased_coin(mp.rng))
        {
            LOG_INFO("generalized_dp_argmax: early unsuccessful exit");
            return;
        }
    }
    LOG_INFO("generalized_dp_argmax: late unsuccessful exit");
    return;
}

void DPEnsemble::init_gradients(DataSet &ds)
{
    std::vector<double> init_scores(ds.length, this->init_score);
    ds.gradients = params->task->compute_gradients(ds.y, init_scores);
}

void DPEnsemble::update_gradients(DataSet &ds)
{
    auto y_pred = this->predict(ds.X);
    ds.gradients = (this->params->task)->compute_gradients(ds.y, y_pred);
}

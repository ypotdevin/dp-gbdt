#include <stdexcept>
#include "estimator.h"

namespace dpgbdt
{
    Estimator::Estimator()
    {
        this->params = std::shared_ptr<ModelParams>(new ModelParams());
        this->params->rng = std::mt19937(std::random_device{}());
        this->params->privacy_budget = 1.0;
        this->params->ensemble_rejector_budget_split = 0.9;
        this->params->tree_rejector = std::shared_ptr<tree_rejection::DPrMSERejector>(new tree_rejection::DPrMSERejector(5, 100.0, 2.0, this->params->rng));
        //this->params->tree_scorer = std::shared_ptr<tree_rejection::DPrMSEScorer>(new tree_rejection::DPrMSEScorer(100.0, 2.0, this->params->rng));
        this->params->tree_scorer = std::shared_ptr<tree_rejection::DPQuantileScorer>(new tree_rejection::DPQuantileScorer(0.0, 1.0, {0.5, 0.90, 0.95}, 100.0, this->params->rng));
        this->params->dp_argmax_privacy_budget = 0.1;
        this->params->stopping_prob = 0.05;
        this->params->learning_rate = 0.1;
        this->params->n_trees_to_accept = 5;
        this->params->max_depth = 6;
        this->params->min_samples_split = 2;
        this->params->l2_threshold = 1.0;
        this->params->l2_lambda = 0.1;
        this->params->balance_partition = true;
        this->params->gradient_filtering = true;
        this->params->leaf_clipping = true;
        this->params->use_decay = false;
        this->params->task = std::shared_ptr<Task>(new Regression());
    }

    Estimator::Estimator(
        std::mt19937 const &rng,
        double privacy_budget,
        double ensemble_rejector_budget_split,
        std::shared_ptr<tree_rejection::TreeRejector> tree_rejector,
        std::shared_ptr<tree_rejection::TreeScorer> tree_scorer,
        double dp_argmax_privacy_budget,
        double dp_argmax_stopping_prob,
        double learning_rate,
        int n_trees_to_accept,
        int max_depth,
        int min_samples_split,
        double l2_threshold,
        double l2_lambda,
        bool balance_partition,
        bool gradient_filtering,
        bool leaf_clipping,
        bool use_decay)
    {
        this->params = std::shared_ptr<ModelParams>(new ModelParams());
        this->params->rng = rng;
        this->params->privacy_budget = privacy_budget;
        this->params->ensemble_rejector_budget_split = ensemble_rejector_budget_split;
        this->params->tree_rejector = tree_rejector;
        this->params->tree_scorer = tree_scorer;
        this->params->dp_argmax_privacy_budget = dp_argmax_privacy_budget;
        this->params->stopping_prob = dp_argmax_stopping_prob;
        this->params->learning_rate = learning_rate;
        this->params->n_trees_to_accept = n_trees_to_accept;
        this->params->max_depth = max_depth;
        this->params->min_samples_split = min_samples_split;
        this->params->l2_threshold = l2_threshold;
        this->params->l2_lambda = l2_lambda;
        this->params->balance_partition = balance_partition;
        this->params->gradient_filtering = gradient_filtering;
        this->params->leaf_clipping = leaf_clipping;
        this->params->use_decay = use_decay;
        this->params->task = std::shared_ptr<Task>(new Regression());
    }

    Estimator &Estimator::fit(std::vector<std::vector<double>> X, std::vector<double> y, std::vector<int> cat_idx, std::vector<int> num_idx)
    {
        /* This might lead to problems, since DPEnsemble takes (?) ownership of
         * passed ModelParams (but does not use shared_pointers instead of raw
         * pointers). */
        this->ensemble = std::shared_ptr<DPEnsemble>(new DPEnsemble(this->params.get()));
        this->params->cat_idx = cat_idx;
        this->params->num_idx = num_idx;
        DataSet dataset = DataSet(X, y);
        this->ensemble->train(&dataset);
        return *this;
    }

    std::vector<double> Estimator::predict(std::vector<std::vector<double>> X)
    {
        if (this->ensemble)
        {
            return this->ensemble->predict(X);
        }
        else
        {
            throw std::runtime_error("Estimator is not fitted.");
        }
    }
}
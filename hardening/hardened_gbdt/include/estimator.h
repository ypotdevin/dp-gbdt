#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include "dp_ensemble.h"
#include "tree_rejection.h"

namespace dpgbdt
{
    /**
     * @brief A minimal, sci-kit learn like interface for accessing
     * differentially private gradient boosted decision trees for regression
     * (not classification).
     */
    class Estimator
    {
    public:
        /**
         * @brief By default, this yields a DP GBDT ensemble with default
         * settings and DP-rMSE-based tree rejection.
         */
        Estimator();

        /**
         * @brief A DP GBDT according to specified parameters.
         *
         * No input validation is performed at this stage. This happens in
         * Estimator::fit.
         *
         * @param rng the random number generator to use for ensemble generation
         * @param privacy_budget the privacy budget for the ensemble
         * @param tree_rejector the tree rejection mechanism to use
         * @param learning_rate the learning rate to use
         * @param nb_trees the maximal size of the ensemble (if all trees will
         * be accepted)
         * @param max_depth the maximal depth per each individual tree
         * @param min_samples_split minimal amount of samples required to split
         * a tree node
         * @param l2_threshold the l2_threshold to use
         * @param l2_lambda the l2_lambda to use
         * @param balance_partition whether to balance the partition
         * @param gradient_filtering whether to perform gradient filtering
         * @param leaf_clipping whether to perform leaf clipping
         * @param use_decay whether to use decay
         */
        Estimator(
            std::mt19937 const &rng,
            double privacy_budget,
            std::shared_ptr<tree_rejection::TreeRejector> tree_rejector,
            double learning_rate,
            int nb_trees,
            int max_depth,
            int min_samples_split,
            double l2_threshold,
            double l2_lambda,
            bool balance_partition,
            bool gradient_filtering,
            bool leaf_clipping,
            bool use_decay);

        /**
         * @brief Perform input validation and fit the DP GBDT to the provided
         * data.
         *
         * @param X the training data input variables
         * @param y the training data output variable
         * @param cat_idx which columns (indices) of X are of categorical
         * nature?
         * @param num_idx which columns (indices) of X are of numerical nature?
         * @return the fitted estimator
         */
        Estimator &fit(std::vector<std::vector<double>> X, std::vector<double> y, std::vector<int> cat_idx, std::vector<int> num_idx);

        /**
         * @brief Infer the output variables corresponding to the provided input
         * variables.
         *
         * Note: It is necessary to fit this estimator before calling predict.
         *
         * @param X input variables (need to be of the same type is the training
         * input variables, especially concerning the positioning of categorical
         * and numerical features).
         * @return std::vector<double> the inferred output variables
         */
        std::vector<double> predict(std::vector<std::vector<double>> X);

    private:
        std::shared_ptr<ModelParams> params;
        std::shared_ptr<DPEnsemble> ensemble;
    };

}

#endif // ESTIMATOR_H
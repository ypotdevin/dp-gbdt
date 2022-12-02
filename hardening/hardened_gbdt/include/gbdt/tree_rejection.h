#ifndef TREE_REJECTION_H
#define TREE_REJECTION_H

#include <memory>
#include <random>
#include <vector>
#include "custom_cauchy.h"
#include "laplace.h"

namespace tree_rejection
{
    class TreeScorer
    {
    public:
        virtual double score_tree(double privacy_budget, const std::vector<double> &y, const std::vector<double> &y_pred) = 0;
    };

    class DPrMSEScorer : public TreeScorer
    {
    private:
        double upper_bound;
        std::unique_ptr<custom_cauchy::CustomCauchy> cc;

    public:
        DPrMSEScorer(double upper_bound, double gamma, const std::mt19937 &rng);
        double score_tree(double privacy_budget, const std::vector<double> &y, const std::vector<double> &y_pred);
    };

    class DPQuantileScorer : public TreeScorer
    {
    private:
        double shift, scale, upper_bound;
        std::vector<double> qs;
        std::mt19937 rng;

    public:
        DPQuantileScorer(double shift, double scale, const std::vector<double> &qs, double upper_bound, std::mt19937 &rng);
        double score_tree(double privacy_budget, const std::vector<double> &y, const std::vector<double> &y_pred);
    };

    /**
     * @brief Use insights gained from "Average-Case Averages: Private
     * Algorithms for Smooth Sensitivity and Mean Estimation", Mark Bun &
     * Thomas Steinke (2019) to replace the Cauchy noise distribution in
     * DPrMSEScorer by a Laplace distribution. Consequences: Only approximate
     * differential privacy (epsilon, delta) is achieved, in contrast to regular
     * differential privacy (just epsilon).
     *
     * Note: The score is still based on rMSE.
     */
    class BunSteinkeScorer : public TreeScorer
    {
    private:
        /* `relaxation` is the delta parameter */
        double upper_bound, beta, relaxation;
        std::unique_ptr<Laplace> std_laplace;

    public:
        BunSteinkeScorer(double upper_bound, double beta, double relaxation, std::mt19937 &rng);
        double score_tree(double privacy_budget, const std::vector<double> &y, const std::vector<double> &y_pred);
    };

    class TreeRejector
    {
    public:
        /**
         * Note: Before calling this method for the first time, call
         * `set_total_privacy_budget` at least once.
         *
         * @param y the target regression values.
         * @param y_pred the ensemble's (including the most recent tree which is
         * going to be judged upon) predicted regression values.
         * @return true reject the most recent tree associated with the
         * prediction `y_pred`.
         * @return false accept the most recent tree associated with the
         * prediction `y_pred`.
         */
        virtual bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred) = 0;
        /**
         * @brief Set the privacy budget the tree rejector is allowed to spend
         * over the whole training period, in total.
         *
         * It is intended that the ambient ensemble sets this value based on
         * internal calculations.
         *
         * Note: Call this method before the first call to reject_tree`.
         */
        virtual void set_total_privacy_budget(double budget) = 0;
        virtual void print(std::ostream &os) const = 0;
        friend std::ostream &operator<<(std::ostream &os, const TreeRejector &tr)
        {
            tr.print(os);
            return os;
        }
    };

    /**
     * @brief Either always keep the considered tree, or always reject the
     * considered tree.
     *
     */
    class ConstantRejector : public TreeRejector
    {
    private:
        bool decision;

    public:
        /**
         * @brief Construct a new Constant Rejector object
         *
         * @param decision whether to always reject (true) or to always accept
         * (false) the considered tree.
         */
        ConstantRejector(bool decision);
        void print(std::ostream &os) const;

        /**
         * @brief Depending on `decision` (provided via initialization), either
         * return always true or always false.
         *
         * @param y ground truth (unused)
         * @param y_pred prediction (unused)
         * @return true if `decision` is true
         * @return false if decision is false
         */
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);

        void set_total_privacy_budget(double budget);
    };

    /**
     * @brief Keep the considered tree, if the approximated (via a quantile)
     * root mean squared error of the ensemble including the tree is strictly
     * lower than the error of the ensemble without the considered tree.
     * Otherwise, reject the considered tree.
     */
    class QuantileRejector : public TreeRejector
    {
    private:
        double previous_error, q;

    public:
        QuantileRejector(double q);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
        void set_total_privacy_budget(double budget);
    };

    /**
     * @brief Keep the considered tree, if the approximated (via a linear
     * combination of quantiles) root mean squared error of the ensemble,
     * including the tree, is strictly lower than the error of the ensemble
     * without the considered tree. Otherwise, reject the considered tree.
     *
     * This rejector generalizes the QuantileCombinationRejector, as the
     * coefficients don't need to be a probability vector.
     *
     */
    class QuantileLinearCombinationRejector : public TreeRejector
    {
    private:
        double previous_error;
        std::vector<double> qs, coefficients;

    public:
        /**
         * @param qs the target quantiles (must have same length as
         * `coefficients`)
         * @param coefficients the corresponding coefficients of the target
         * quantiles (must have same length as `qs`).
         */
        QuantileLinearCombinationRejector(std::vector<double> qs, std::vector<double> coefficients);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
        void set_total_privacy_budget(double budget);
    };

    /**
     * @brief Keep the considered tree, if the approximated (via a convex
     * combination of quantiles) root mean squared error of the ensemble
     * including the tree is strictly lower than the error of the ensemble
     * without the considered tree. Otherwise, reject the considered tree.
     *
     */
    class QuantileCombinationRejector : public TreeRejector
    {
    private:
        std::vector<double> qs, weights;
        std::shared_ptr<QuantileLinearCombinationRejector> qlcr;

    public:
        /**
         * @param qs the target quantiles (must have same length as `weights`)
         * @param weights the corresponding weights of the target quantiles
         * (must have same length as `qs`). The weights may sum up to 1.0 or
         * may be expressed as non-negative, relative weights. They will be
         * normalized anyway.
         */
        QuantileCombinationRejector(std::vector<double> qs, std::vector<double> weights);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
        void set_total_privacy_budget(double budget);
    };

    /**
     * @brief Keep the considered tree, if the approximated (via our
     * differentially private root mean squared error estimation) error of the
     * ensemble including the tree is strictly lower than the error of the
     * ensemble without the considered tree. Otherwise, reject the considered
     * tree.
     */
    class DPrMSERejector : public TreeRejector
    {
    private:
        double epsilon, U, previous_error;
        int n_trees_to_accept, n_accepted_trees;
        std::unique_ptr<custom_cauchy::CustomCauchy> cc;

    public:
        /**
         * @brief Construct a new DPrMSERejector object
         *
         * @param n_trees_to_accept how many trees to accept at most (upper
         * bound).
         * @param U the upper bound on absolute errors (of regression values and
         * predicted regression values). Required for smooth sensitivity
         * calculation.
         * @param gamma the gamma to use for the custom Cauchy noise
         * distribution.
         * @param rng the random number generator used by the custom Cauchy
         * distribution.
         */
        DPrMSERejector(int n_trees_to_accept, double U, double gamma, const std::mt19937 &rng);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
        void set_total_privacy_budget(double budget);
    };

    class ApproxDPrMSERejector : public TreeRejector
    {
    private:
        double epsilon, delta, U, previous_error;
        int n_trees_to_accept, n_accepted_trees;
        std::unique_ptr<Laplace> laplace_distr;

    public:
        ApproxDPrMSERejector(int n_trees_to_accept, double delta, double U, std::mt19937 &rng);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
        void set_total_privacy_budget(double budget);
    };
}
#endif /* TREE_REJECTION_H */
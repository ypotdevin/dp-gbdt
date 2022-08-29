#ifndef TREE_REJECTION_H
#define TREE_REJECTION_H

#include <memory>
#include <random>
#include <vector>
#include "cli_parser.h"
#include "custom_cauchy.h"
#include "laplace.h"

namespace tree_rejection
{
    class TreeRejector
    {
    public:
        virtual bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred) = 0;
        virtual void print(std::ostream &os) const = 0;
        friend std::ostream &operator<<(std::ostream &os, const TreeRejector &tr)
        {
            tr.print(os);
            return os;
        }
    };

    /**
     * @brief Depending on the passed command line arguments, create the
     * demanded tree rejector and initialize it with the given parameters.
     *
     * @param cp the parser containing the command line arguments.
     * @param rng the random number generator used by this program, eventually
     * needed by the tree rejector.
     */
    std::shared_ptr<TreeRejector> from_CommandLineParser(cli_parser::CommandLineParser &cp, std::mt19937 &rng);

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
    };

    /**
     * @brief Keep the considered tree, if the approximated (via our
     * differentially private root mean squared error estimation) error of the
     * ensemble including the tree is strictly lower than the error of the
     * ensemble without the considered tree. Otherwise, reject the considered
     * tree.
     *
     */
    class DPrMSERejector : public TreeRejector
    {
    private:
        double epsilon, U, previous_error;
        std::unique_ptr<custom_cauchy::CustomCauchy> cc;

    public:
        DPrMSERejector(double epsilon, double U, double gamma, const std::mt19937 &rng);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
    };

    class ApproxDPrMSERejector : public TreeRejector
    {
    private:
        double epsilon, delta, U, previous_error;
        std::unique_ptr<Laplace> laplace_distr;

    public:
        ApproxDPrMSERejector(double epsilon, double delta, double U, std::mt19937 &rng);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
    };
}
#endif /* TREE_REJECTION_H */
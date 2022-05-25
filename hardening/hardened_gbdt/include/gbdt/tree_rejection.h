#ifndef TREE_REJECTION_H
#define TREE_REJECTION_H

#include <memory>
#include <random>
#include <vector>
#include "cli_parser.h"
#include "custom_cauchy.h"

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
    std::unique_ptr<TreeRejector> from_CommandLineParser(cli_parser::CommandLineParser &cp, const std::mt19937_64 &rng);

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
     * @brief Keep the considered tree, if the approximated (via a convex
     * combination of quantiles) root mean squared error of the ensemble
     * including the tree is strictly lower than the error of the ensemble
     * without the considered tree. Otherwise, reject the considered tree.
     *
     */
    class QuantileCombinationRejector : public TreeRejector
    {
    private:
        double previous_error;
        std::vector<double> qs, weights;

    public:
        /**
         * @brief Construct a new Quantile Combination Rejector object
         *
         * @param qs the target quantiles (must have same length as `weights`)
         * @param weights the corresponding weights of the target quantiles
         * (must have same length as `qs`).
         */
        QuantileCombinationRejector(const std::vector<double> qs, const std::vector<double> weights);
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
        DPrMSERejector(double epsilon, double U, double gamma, const std::mt19937_64 &rng);
        void print(std::ostream &os) const;
        bool reject_tree(std::vector<double> &y, std::vector<double> &y_pred);
    };
}
#endif /* TREE_REJECTION_H */
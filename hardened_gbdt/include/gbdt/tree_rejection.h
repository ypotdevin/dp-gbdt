#ifndef TREE_REJECTION_H
#define TREE_REJECTION_H

#include <memory>
#include <random>
#include <vector>
#include "custom_cauchy.h"
#include "laplace.h"

namespace tree_rejection
{
    class Beta
    {
    public:
        virtual double beta(double privacy_budget, double relaxation) = 0;
    };

    class ConstantBeta : public Beta
    {
    private:
        double beta_constant;

    public:
        ConstantBeta(double beta);
        double beta(double privacy_budget, double relaxation);
    };

    class StandardCauchyBeta : public Beta
    {
    public:
        double beta(double privacy_budget, double relaxation);
    };

    class CustomCauchyBeta : public Beta
    {
    private:
        double gamma;

    public:
        CustomCauchyBeta(double gamma);
        double beta(double privacy_budget, double relaxation);
    };

    class TreeScorer
    {
    public:
        virtual double score_tree(double privacy_budget,
                                  const std::vector<double> &y,
                                  const std::vector<double> &y_pred) = 0;
    };

    class LeakyRmseScorer : public TreeScorer
    {
    public:
        LeakyRmseScorer();
        double score_tree(double privacy_budget,
                          const std::vector<double> &y,
                          const std::vector<double> &y_pred);
    };

    class DPrMSEScorer2 : public TreeScorer
    {
    private:
        std::shared_ptr<Beta> beta;
        double upper_bound;
        std::unique_ptr<custom_cauchy::CustomCauchy> cc;

    public:
        DPrMSEScorer2(
            std::shared_ptr<Beta> beta_ptr,
            double upper_bound,
            double gamma,
            const std::mt19937 &rng);
        double score_tree(double privacy_budget,
                          const std::vector<double> &y,
                          const std::vector<double> &y_pred);
    };

    class DPrMSEScorer : public TreeScorer
    {
    private:
        DPrMSEScorer2 scorer;

    public:
        DPrMSEScorer(double upper_bound, double gamma, const std::mt19937 &rng);
        double score_tree(double privacy_budget,
                          const std::vector<double> &y,
                          const std::vector<double> &y_pred);
    };

    class DPQuantileScorer : public TreeScorer
    {
    private:
        double shift, scale, upper_bound;
        std::vector<double> qs;
        std::mt19937 rng;

    public:
        DPQuantileScorer(double shift,
                         double scale,
                         const std::vector<double> &qs,
                         double upper_bound,
                         std::mt19937 &rng);
        double score_tree(double privacy_budget,
                          const std::vector<double> &y,
                          const std::vector<double> &y_pred);
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
        BunSteinkeScorer(double upper_bound,
                         double beta,
                         double relaxation,
                         std::mt19937 &rng);
        double score_tree(double privacy_budget,
                          const std::vector<double> &y,
                          const std::vector<double> &y_pred);
    };

    class PrivacyBucketScorer : public TreeScorer
    {
    private:
        double upper_bound, beta;
        int n_trees_to_accept;
        std::vector<double> coefficients;
        std::mt19937 rng;
        std::unique_ptr<std::normal_distribution<double>> std_gaussian;

    public:
        PrivacyBucketScorer(double upper_bound,
                            double beta,
                            int n_trees_to_accept,
                            const std::vector<double> &coefficients,
                            std::mt19937 &rng);
        double score_tree(double privacy_budget,
                          const std::vector<double> &y,
                          const std::vector<double> &y_pred);
    };
}
#endif /* TREE_REJECTION_H */
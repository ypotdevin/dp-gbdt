#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <random>
#include <sstream>

#include "gbdt/tree_rejection.h"
#include "gbdt/loss.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "gbdt/utils.h"

// Some "private" helpers
namespace
{
    /**
     * @param q
     * @param lower_bound
     * @param values assumed to be sorted!
     * @param upper_bound
     * @param privacy_budget
     * @param rng
     * @return double a DP approximate of the q-quantile of values
     */
    double exp_q(double q, double lower_bound, const std::vector<double> &values, double upper_bound, double privacy_budget, std::mt19937 &rng)
    {
        auto samples = values;
        samples.insert(samples.begin(), lower_bound);
        samples.push_back(upper_bound);
        std::size_t n = samples.size() - 1;
        std::size_t m = std::floor((n + 1) * q); // this prefers the lower quantile
        std::vector<double> probs(n);
        for (std::size_t i = 0; i < m; ++i)
        {
            double width = samples.at(i + 1) - samples.at(i);
            double utility = (i + 1) - m;
            double prob = width * std::exp(privacy_budget * utility / 2.0);
            probs.at(i) = std::max(prob, 0.0);
        }
        for (std::size_t i = m; i < n; i++)
        {
            double width = samples.at(i + 1) - samples.at(i);
            double utility = m - i;
            double prob = width * std::exp(privacy_budget * utility / 2.0);
            probs.at(i) = std::max(prob, 0.0);
        }
        auto dp_q_index = numpy::choice(probs, rng);
        std::uniform_real_distribution<double> bin(samples.at(dp_q_index), samples.at(dp_q_index + 1));
        auto dp_q = bin(rng);
        return dp_q;
    }
}

namespace tree_rejection
{
    ConstantBeta::ConstantBeta(double beta) : beta_constant(beta)
    {
    }

    double ConstantBeta::beta(double privacy_budget, double relaxation)
    {
        return this->beta_constant;
    }

    double StandardCauchyBeta::beta(double privacy_budget, double relaxation)
    {
        return privacy_budget / 6.0;
    }

    CustomCauchyBeta::CustomCauchyBeta(double gamma) : gamma(gamma)
    {
        if (gamma <= 1.0)
        {
            throw std::runtime_error("Gamma needs to be larger than 1.0");
        }
    }

    double CustomCauchyBeta::beta(double privacy_budget, double relaxation)
    {
        return privacy_budget / (2 * (this->gamma + 1.0));
    }

    LeakyRmseScorer::LeakyRmseScorer() {}

    double LeakyRmseScorer::score_tree(double privacy_budget,
                                       const std::vector<double> &y,
                                       const std::vector<double> &y_pred)
    {
        return compute_rmse(absolute_differences(y, y_pred));
    }

    DPrMSEScorer::DPrMSEScorer(
        double upper_bound,
        double gamma,
        const std::mt19937 &rng)
        : scorer(std::shared_ptr<CustomCauchyBeta>(new CustomCauchyBeta(gamma)),
                 upper_bound,
                 gamma,
                 rng)
    {
    }

    double DPrMSEScorer::score_tree(double privacy_budget,
                                    const std::vector<double> &y,
                                    const std::vector<double> &y_pred)
    {
        return this->scorer.score_tree(privacy_budget, y, y_pred);
    }

    DPrMSEScorer2::DPrMSEScorer2(
        std::shared_ptr<Beta> beta_ptr,
        double upper_bound,
        double gamma,
        const std::mt19937 &rng)
        : beta(beta_ptr),
          upper_bound(upper_bound),
          cc(std::unique_ptr<custom_cauchy::AdvancedCustomCauchy>(
              new custom_cauchy::AdvancedCustomCauchy(gamma, rng)))
    {
        if (upper_bound < 0.0 || gamma <= 1.0)
        {
            throw std::runtime_error("DPrMSEScorer2: upper_bound < 0 or gamma <= 1");
        }
    }

    double DPrMSEScorer2::score_tree(double privacy_budget,
                                     const std::vector<double> &y,
                                     const std::vector<double> &y_pred)
    {
        auto abs_errors = absolute_differences(y, y_pred);
        auto score = dp_rms_custom_cauchy(abs_errors,
                                          privacy_budget,
                                          this->beta->beta(privacy_budget, 0.0),
                                          this->upper_bound,
                                          *this->cc);
        LOG_INFO("### diagnosis value 02 ### - rmse_approx={1}", score);
        return score;
    }

    DPQuantileScorer::DPQuantileScorer(double shift, double scale, const std::vector<double> &qs, double upper_bound, std::mt19937 &rng)
    {
        this->shift = shift;
        this->scale = scale;
        this->qs = qs;
        this->upper_bound = upper_bound;
        this->rng = rng;
    }

    double DPQuantileScorer::score_tree(double privacy_budget, const std::vector<double> &y, const std::vector<double> &y_pred)
    {
        std::vector<double> abs_errors(y.size());
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), abs_errors.begin(), [](double _y, double _y_pred)
                       { return std::abs(_y - _y_pred); });
        std::sort(abs_errors.begin(), abs_errors.end());

        std::size_t n = this->qs.size();
        std::vector<double> quantiles;
        auto per_quantile_budget = privacy_budget / n;
        for (auto q : this->qs)
        {
            quantiles.push_back(
                exp_q(q, 0.0, abs_errors, this->upper_bound, per_quantile_budget, this->rng));
        }

        auto qs_ = this->qs;
        qs_.insert(qs_.begin(), 0.0);
        qs_.push_back(1.0);
        quantiles.push_back(quantiles.back());
        double sum = 0.0;
        for (std::size_t i = 1; i < qs_.size(); ++i)
        {
            sum += (qs_.at(i) - qs_.at(i - 1)) * std::pow(quantiles.at(i - 1), 2.0);
        }
        auto score = this->shift + this->scale * std::sqrt(sum);
        return score;
    }

    BunSteinkeScorer::BunSteinkeScorer(double upper_bound,
                                       double beta,
                                       double relaxation,
                                       std::mt19937 &rng)
    {
        this->upper_bound = upper_bound;
        if (beta < 0.0)
        {
            throw std::runtime_error("Parameter beta must be not negative!");
        }
        this->beta = beta;
        this->relaxation = relaxation;
        this->std_laplace = std::unique_ptr<Laplace>(new Laplace(rng));
    }

    double BunSteinkeScorer::score_tree(double privacy_budget,
                                        const std::vector<double> &y,
                                        const std::vector<double> &y_pred)
    {
        LOG_INFO("Calculating beta-smooth sensitivity for beta={1}", this->beta);
        auto alpha = privacy_budget + this->beta -
                     (std::exp(this->beta) - 1.0) * std::log(1.0 / this->relaxation);
        LOG_INFO("### diagnosis value 17 ### alpha={1}", alpha);
        if (alpha <= 0.0)
        {
            return std::nan(""); /** TODO: throw a runtime error? */
        }
        std::vector<double> abs_errors(y.size());
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), abs_errors.begin(), [](double _y, double _y_pred)
                       { return std::abs(_y - _y_pred); });
        std::sort(abs_errors.begin(), abs_errors.end());
        double smooth_sens, rmse;
        std::tie(smooth_sens, rmse) = rMS_smooth_sensitivity(abs_errors,
                                                             this->beta,
                                                             this->upper_bound);
        auto noise = this->std_laplace->return_a_random_variable();
        auto score = rmse + noise * smooth_sens / alpha;
        return score;
    }

    PrivacyBucketScorer::PrivacyBucketScorer(double upper_bound,
                                             double beta,
                                             int n_trees_to_accept,
                                             const std::vector<double> &coefficients,
                                             std::mt19937 &rng)
    {
        this->upper_bound = upper_bound;
        if (beta < 0.0)
        {
            throw std::runtime_error("Parameter beta must be not negative!");
        }
        this->beta = beta;
        this->n_trees_to_accept = n_trees_to_accept;
        this->coefficients = coefficients;
        this->rng = rng;
        this->std_gaussian = std::unique_ptr<std::normal_distribution<double>>(
            new std::normal_distribution<double>());
    }

    double PrivacyBucketScorer::score_tree(double privacy_budget,
                                           const std::vector<double> &y,
                                           const std::vector<double> &y_pred)
    {
        auto total_score_budget = privacy_budget * this->n_trees_to_accept;
        auto sfc = poly2d(total_score_budget, this->beta, this->coefficients);
        LOG_INFO("### diagnosis value 19 ### sfc={1}, total_score_budget={2}, beta={3}", sfc, total_score_budget, this->beta);
        std::vector<double> abs_errors(y.size());
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), abs_errors.begin(), [](double _y, double _y_pred)
                       { return std::abs(_y - _y_pred); });
        std::sort(abs_errors.begin(), abs_errors.end());
        double smooth_sens, rmse;
        std::tie(smooth_sens, rmse) = rMS_smooth_sensitivity(abs_errors,
                                                             this->beta,
                                                             this->upper_bound);
        auto noise = this->std_gaussian->operator()(this->rng);
        auto loose_gaussian_bound = 5.3;
        auto scaling_factor = loose_gaussian_bound * std::sqrt(this->n_trees_to_accept) / sfc;
        auto score = rmse + smooth_sens * scaling_factor * noise;
        return score;
    }
}
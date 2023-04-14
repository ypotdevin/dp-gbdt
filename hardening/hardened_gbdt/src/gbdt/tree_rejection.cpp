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
    std::vector<double> quantiles(std::vector<double> samples, std::vector<double> qs)
    {
        std::vector<double> quants;
        auto n = samples.size() - 1;
        for (auto q : qs)
        {
            std::size_t quantile_position = std::ceil(q * n);
            std::nth_element(samples.begin(), samples.begin() + quantile_position, samples.end());
            quants.push_back(samples.at(quantile_position));
        }

        return quants;
    }

    /**
     * @brief Compute the q-th quantile of the given sample. If q doesn't match
     * exactly an index, ceil for the next-larger element.
     *
     * @param samples the samples to compute the quantile for. It is not
     * necessary to sort the samples previously.
     * @param q
     * @return double
     */
    double quantile(std::vector<double> samples, double q)
    {
        std::vector<double> quants = {q};
        return quantiles(samples, quants)[0];
    }

    std::string dvec2listrepr(std::vector<double> vec)
    {
        if (vec.empty())
        {
            return "[]";
        }
        else
        {
            std::ostringstream oss;
            oss << "[";
            std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<double>(oss, ", "));
            oss << vec.back()
                << "]";
            return oss.str();
        }
    }

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

    ConstantRejector::ConstantRejector(bool decision)
    {
        this->decision = decision;
    }

    void ConstantRejector::print(std::ostream &os) const
    {
        os << "\"ConstantRejector(decision=" << this->decision << "\")";
    }

    bool ConstantRejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        return decision;
    }

    void ConstantRejector::set_total_privacy_budget(double budget){};

    QuantileRejector::QuantileRejector(double q)
    {
        this->q = q;
        this->previous_error = std::numeric_limits<double>::max();
    }

    void QuantileRejector::print(std::ostream &os) const
    {
        os << "\"QuantileRejector(q=" << this->q << ")\"";
    }

    bool QuantileRejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), std::minus<double>());

        auto current_error = quantile(y_pred, q);
        if (current_error < previous_error)
        {
            previous_error = current_error;
            return false; // do not reject
        }
        else
        {
            return true; // reject
        }
    }

    void QuantileRejector::set_total_privacy_budget(double budget){};

    QuantileCombinationRejector::QuantileCombinationRejector(std::vector<double> qs, std::vector<double> weights)
    {
        this->qs = qs;
        this->weights = weights;
        normalize(this->weights);
        this->qlcr = std::shared_ptr<QuantileLinearCombinationRejector>(new QuantileLinearCombinationRejector(this->qs, this->weights));
    }

    void QuantileCombinationRejector::print(std::ostream &os) const
    {
        os << "\"QuantileCombinationRejector(qs="
           << dvec2listrepr(this->qs)
           << ",weights="
           << dvec2listrepr(this->weights)
           << ")\"";
    }

    bool QuantileCombinationRejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        return this->qlcr->reject_tree(y, y_pred);
    }

    void QuantileCombinationRejector::set_total_privacy_budget(double budget){};

    QuantileLinearCombinationRejector::QuantileLinearCombinationRejector(std::vector<double> qs, std::vector<double> coefficients)
    {
        this->previous_error = std::numeric_limits<double>::max();
        this->qs = qs;
        this->coefficients = coefficients;
    }

    void QuantileLinearCombinationRejector::print(std::ostream &os) const
    {
        os << "\"QuantileLinearCombinationRejector(qs="
           << dvec2listrepr(this->qs)
           << ",coefficients="
           << dvec2listrepr(this->coefficients)
           << ")\"";
    }

    bool QuantileLinearCombinationRejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        /* Compute absolute errors, so the quantiles make sense. Although this
         * is not necessary for rMSE calculation (since the errors will be
         * squared anyway), this might be beneficial for the estimation via
         * quantiles.
         */
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), [](double _y, double _y_pred)
                       { return std::abs(_y - _y_pred); });
        auto diagnosis_quants = dvec2listrepr(quantiles(y_pred, numpy::linspace(0.5, 1.0, 0.05))); // [0.50, 0.55, …, 0.95, 1.0]
        LOG_INFO("### diagnosis value 01 ### - rmse={1}", compute_rmse(y_pred));
        LOG_INFO("### diagnosis value 03 ### - quantiles={1}", diagnosis_quants);

        double current_error = 0.0;
        auto quants = quantiles(y_pred, this->qs);
        current_error = std::inner_product(this->coefficients.begin(), this->coefficients.end(), quants.begin(), current_error);

        LOG_INFO("### diagnosis value 02 ### - rmse_approx={1}", current_error);
        if (current_error < previous_error)
        {
            previous_error = current_error;
            return false; // do not reject
        }
        else
        {
            return true; // reject
        }
    }

    void QuantileLinearCombinationRejector::set_total_privacy_budget(double budget){};

    DPrMSERejector::DPrMSERejector(int n_trees_to_accept, double U, double gamma, const std::mt19937 &rng)
    {
        this->epsilon = -1.0;
        this->U = U;
        this->n_trees_to_accept = n_trees_to_accept;
        this->n_accepted_trees = 0;
        this->cc = std::unique_ptr<custom_cauchy::AdvancedCustomCauchy>(new custom_cauchy::AdvancedCustomCauchy(gamma, rng));
        this->previous_error = std::numeric_limits<double>::max();
    }

    void DPrMSERejector::print(std::ostream &os) const
    {
        os << "\"DPrMSERejector(per_call_eps="
           << this->epsilon
           << ",n_trees_to_accept="
           << this->n_trees_to_accept
           << ",U="
           << this->U
           << ",custom_cauchy="
           << *this->cc
           << ")\"";
    }

    bool DPrMSERejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        if (this->epsilon <= 0.0)
        {
            throw std::runtime_error(
                "Insufficient privacy budget provided. Was set_total_privacy_budget(...) called before?");
        }
        if (this->n_accepted_trees >= this->n_trees_to_accept)
        {
            LOG_INFO(
                "Likely unintended behavior detected: Calling reject_tree(...) again, even after accepting {1} of {2} trees.",
                this->n_accepted_trees,
                this->n_trees_to_accept);
        }
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), [](double _y, double _y_pred)
                       { return std::abs(_y - _y_pred); });
        LOG_INFO("### diagnosis value 01 ### - rmse={1}", compute_rmse(y_pred));
        auto current_error = dp_rms_custom_cauchy(y_pred, this->epsilon, this->U, *this->cc);
        LOG_INFO("### diagnosis value 02 ### - rmse_approx={1}", current_error);
        if (current_error < this->previous_error)
        {
            this->previous_error = current_error;
            this->n_accepted_trees += 1;
            return false; // do not reject
        }
        else
        {
            return true; // reject
        }
    }

    void DPrMSERejector::set_total_privacy_budget(double budget)
    {
        this->epsilon = budget / this->n_trees_to_accept;
    }

    ApproxDPrMSERejector::ApproxDPrMSERejector(int n_trees_to_accept, double delta, double U, std::mt19937 &rng)
    {
        this->epsilon = -1.0;
        this->delta = delta;
        this->U = U;
        this->n_trees_to_accept = n_trees_to_accept;
        this->n_accepted_trees = 0;
        this->previous_error = std::numeric_limits<double>::max();
        std::uniform_int_distribution<int> uni(0, std::numeric_limits<int>::max());
        auto seed = uni(rng);
        this->laplace_distr = std::unique_ptr<Laplace>(new Laplace(seed));
    }

    void ApproxDPrMSERejector::print(std::ostream &os) const
    {
        os << "\"ApproxDPrMSERejector(per_call_eps="
           << this->epsilon
           << ",n_trees_to_accept="
           << this->n_trees_to_accept
           << ",delta="
           << this->delta
           << ",U="
           << this->U
           << ")\"";
    }

    bool ApproxDPrMSERejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        if (this->epsilon <= 0.0)
        {
            throw std::runtime_error(
                "Insufficient privacy budget provided. Was set_total_privacy_budget(...) called before?");
        }
        if (this->n_accepted_trees >= this->n_trees_to_accept)
        {
            LOG_INFO(
                "Likely unintended behavior detected: Calling reject_tree(...) again, even after accepting {1} of {2} trees.",
                this->n_accepted_trees,
                this->n_trees_to_accept);
        }
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), [](double _y, double _y_pred)
                       { return std::abs(_y - _y_pred); });

        std::sort(y_pred.begin(), y_pred.end());
        auto beta = this->epsilon / (2.0 * log(2.0 / this->delta));
        double sens, rmse;
        std::tie(sens, rmse) = rMS_smooth_sensitivity(y_pred, beta, this->U);
        LOG_INFO("### diagnosis value 01 ### - rmse={1}", rmse);
        auto noise = this->laplace_distr->return_a_random_variable();
        auto current_error = rmse + 2 * sens * noise / this->epsilon;
        LOG_INFO("### diagnosis value 02 ### - rmse_approx={1}", current_error);
        if (current_error < this->previous_error)
        {
            this->previous_error = current_error;
            return false; // do not reject
        }
        else
        {
            return true; // reject
        }
    }

    void ApproxDPrMSERejector::set_total_privacy_budget(double budget)
    {
        this->epsilon = budget / this->n_trees_to_accept;
    }
}
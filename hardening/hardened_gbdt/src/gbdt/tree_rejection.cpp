#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>

#include "tree_rejection.h"
#include "loss.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "utils.h"

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
     * @param samples
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
}

namespace tree_rejection
{
    std::unique_ptr<TreeRejector> from_CommandLineParser(cli_parser::CommandLineParser &cp, std::mt19937 &rng)
    {
        std::unique_ptr<TreeRejector> tr;
        if (cp.hasOption("--no-tree-rejection"))
        {
            tr = std::unique_ptr<ConstantRejector>(new ConstantRejector(false));
        }
        else if (cp.hasOption("--dp-rmse-tree-rejection"))
        {
            if (cp.hasOption("--rejection-budget") && cp.hasOption("--error-upper-bound") && cp.hasOption("--dp-rmse-gamma"))
            {
                auto epsilon = cp.getDoubleOptionValue("--rejection-budget");
                auto U = cp.getDoubleOptionValue("--error-upper-bound");
                auto gamma = cp.getDoubleOptionValue("--dp-rmse-gamma");
                tr = std::unique_ptr<DPrMSERejector>(new DPrMSERejector(epsilon, U, gamma, rng));
            }
            else
            {
                throw std::runtime_error("Some arguments necessary for DP rMSE tree rejection (via Cauchy) are missing.");
            }
        }
        else if (cp.hasOption("--quantile-rejection"))
        {
            if (cp.hasOption("--quantile-rejection-q"))
            {
                auto q = cp.getDoubleOptionValue("--quantile-rejection-q");
                tr = std::unique_ptr<QuantileRejector>(new QuantileRejector(q));
            }
            else
            {
                throw std::runtime_error("Argument q for quantile tree rejection is missing.");
            }
        }
        else if (cp.hasOption("--quantile-combination-rejection"))
        {
            std::string qcr = "--quantile-combination-rejection";
            std::vector<double> qs, ws;
            for (size_t i = 0; i <= 4; ++i) // Support up to 5 different qs and ws
            {
                auto suffix = std::to_string(i);
                if (cp.hasOption(qcr + "-q" + suffix) && cp.hasOption(qcr + "-w" + suffix))
                {
                    qs.push_back(cp.getDoubleOptionValue(qcr + "-q" + suffix));
                    ws.push_back(cp.getDoubleOptionValue(qcr + "-w" + suffix));
                }
                tr = std::unique_ptr<QuantileCombinationRejector>(new QuantileCombinationRejector(qs, ws));
            }
        }
        else if (cp.hasOption("--dp-laplace-rmse-rejection"))
        {
            if (cp.hasOption("--rejection-budget") && cp.hasOption("--rejection-failure-prob") && cp.hasOption("--error-upper-bound"))
            {
                auto epsilon = cp.getDoubleOptionValue("--rejection-budget");
                auto delta = cp.getDoubleOptionValue("--rejection-failure-prob");
                auto U = cp.getDoubleOptionValue("--error-upper-bound");
                tr = std::unique_ptr<ApproxDPrMSERejector>(new ApproxDPrMSERejector(epsilon, delta, U, rng));
            }
            else
            {
                throw std::runtime_error("Some arguments necessary for approx. DP rMSE tree rejection (via Laplace) are missing.");
            }
        }
        else
        {
            throw std::runtime_error("Selected tree rejection mechanism is unknown.");
        }
        return tr;
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
            return false;
        }
        else
        {
            return true;
        }
    }

    QuantileCombinationRejector::QuantileCombinationRejector(const std::vector<double> qs, const std::vector<double> weights)
    {
        this->qs = qs;
        this->weights = weights;
        normalize(this->weights);
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
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), std::minus<double>());
        std::sort(y_pred.begin(), y_pred.end());

        auto quants = dvec2listrepr(quantiles(y_pred, linspace(0.5, 1.0, 11))); // [0.50, 0.55, …, 0.95, 1.0]
        LOG_INFO("### diagnosis value 01 ### - rmse={1}", compute_rmse(y_pred));
        LOG_INFO("### diagnosis value 03 ### - quantiles={1}", quants);

        auto n = y_pred.size() - 1;
        double current_error = 0.0;
        for (size_t i = 0; i < this->qs.size(); ++i)
        {
            std::size_t quantile_position = std::ceil(this->qs.at(i) * n);
            current_error += this->weights.at(i) * y_pred.at(quantile_position);
        }
        LOG_INFO("### diagnosis value 02 ### - rmse_approx={1}", current_error);
        if (current_error < previous_error)
        {
            previous_error = current_error;
            return false;
        }
        else
        {
            return true;
        }
    }

    DPrMSERejector::DPrMSERejector(double epsilon, double U, double gamma, const std::mt19937 &rng)
    {
        this->epsilon = epsilon;
        this->U = U;
        this->cc = std::unique_ptr<custom_cauchy::AdvancedCustomCauchy>(new custom_cauchy::AdvancedCustomCauchy(gamma, rng));
        this->previous_error = std::numeric_limits<double>::max();
    }

    void DPrMSERejector::print(std::ostream &os) const
    {
        os << "\"DPrMSERejector(eps="
           << this->epsilon
           << ",U="
           << this->U
           << ",custom_cauchy="
           << *this->cc
           << ")\"";
    }

    bool DPrMSERejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), std::minus<double>());
        LOG_INFO("### diagnosis value 01 ### - rmse={1}", compute_rmse(y_pred));
        auto current_error = dp_rms_custom_cauchy(y_pred, this->epsilon, this->U, *this->cc);
        LOG_INFO("### diagnosis value 02 ### - rmse-approx={1}", current_error);
        if (current_error < this->previous_error)
        {
            this->previous_error = current_error;
            return false;
        }
        else
        {
            return true;
        }
    }

    ApproxDPrMSERejector::ApproxDPrMSERejector(double epsilon, double delta, double U, std::mt19937 &rng)
    {
        this->epsilon = epsilon;
        this->delta = delta;
        this->U = U;
        std::uniform_int_distribution<int> uni(0, std::numeric_limits<int>::max());
        auto seed = uni(rng);
        this->laplace_distr = std::unique_ptr<Laplace>(new Laplace(seed));
    }

    void ApproxDPrMSERejector::print(std::ostream &os) const
    {
        os << "\"ApproxDPrMSERejector(eps="
           << this->epsilon
           << ",delta="
           << this->delta
           << ",U="
           << this->U
           << ")\"";
    }

    bool ApproxDPrMSERejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), std::minus<double>());
        LOG_INFO("### diagnosis value 01 ### - rmse={1}", compute_rmse(y_pred));

        std::sort(y_pred.begin(), y_pred.end());
        auto beta = this->epsilon / (2.0 * log(2.0 / this->delta));
        double sens, rmse;
        std::tie(sens, rmse) = rMS_smooth_sensitivity(y_pred, beta, this->U);
        LOG_INFO("#sensitivity_evolution# --- smooth_sens={1}", sens);
        auto noise = this->laplace_distr->return_a_random_variable();
        auto current_error = rmse + 2 * sens * noise / this->epsilon;
        LOG_INFO("### diagnosis value 02 ### - rmse-approx={1}", current_error);
        if (current_error < this->previous_error)
        {
            this->previous_error = current_error;
            return false;
        }
        else
        {
            return true;
        }
    }
}
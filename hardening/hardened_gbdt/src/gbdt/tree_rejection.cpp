#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>

#include "tree_rejection.h"
#include "loss.h"
#include "utils.h"

namespace tree_rejection
{
    std::unique_ptr<TreeRejector> from_CommandLineParser(cli_parser::CommandLineParser &cp, const std::mt19937_64 &rng)
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
                auto budget = cp.getDoubleOptionValue("--rejection-budget");
                auto U = cp.getDoubleOptionValue("--error-upper-bound");
                auto gamma = cp.getDoubleOptionValue("--dp-rmse-gamma");
                tr = std::unique_ptr<DPrMSERejector>(new DPrMSERejector(budget, U, gamma, rng));
            }
            else
            {
                throw std::runtime_error("Some arguments necessary for DP rMSE tree rejection are missing.");
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
        auto n = samples.size() - 1;
        std::size_t quantile_position = std::ceil(q * n);
        std::nth_element(samples.begin(), samples.begin() + quantile_position, samples.end());
        return samples.at(quantile_position);
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

        auto n = y_pred.size() - 1;
        double current_error = 0.0;
        for (size_t i = 0; i < this->qs.size(); ++i)
        {
            std::size_t quantile_position = std::ceil(this->qs.at(i) * n);
            current_error += this->weights.at(i) * y_pred.at(quantile_position);
        }
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

    DPrMSERejector::DPrMSERejector(double epsilon, double U, double gamma, const std::mt19937_64 &rng)
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
        auto current_error = dp_rms_custom_cauchy(y_pred, this->epsilon, this->U, *this->cc);
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
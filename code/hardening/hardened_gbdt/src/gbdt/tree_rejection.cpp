#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <ostream>
#include <sstream>

#include "loss.h"
#include "tree_rejection.h"

namespace tree_rejection
{
    ConstantRejector::ConstantRejector(bool decision)
    {
        this->decision = decision;
    }

    void ConstantRejector::print(std::ostream &os) const
    {
        os << "ConstantRejector(decision=" << this->decision << ")";
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
        auto n = samples.size();
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
        os << "QuantileRejector(q=" << this->q << ")";
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
        os << "QuantileCombinationRejector(qs=["
           << dvec2listrepr(this->qs)
           << "],weights=["
           << dvec2listrepr(this->weights)
           << "])";
    }

    bool QuantileCombinationRejector::reject_tree(std::vector<double> &y, std::vector<double> &y_pred)
    {
        std::transform(y.begin(), y.end(),
                       y_pred.begin(), y_pred.begin(), std::minus<double>());
        std::sort(y_pred.begin(), y_pred.end());

        auto n = y_pred.size();
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
        os << "DPrMSERejector(eps="
           << this->epsilon
           << ",U="
           << this->U
           << ",custom_cauchy="
           << *this->cc
           << ")";
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
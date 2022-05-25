#include <ostream>

#include "custom_cauchy.h"

namespace custom_cauchy
{

    NaiveCustomCauchy::NaiveCustomCauchy(const std::vector<double> standard_support, double gamma, const std::mt19937_64 &rng)
    {
        this->gamma = gamma;
        this->standard_support = standard_support;
        this->rng = rng;
        auto densities = pdf(standard_support, gamma);
        auto normalization_factor = std::accumulate(densities.begin(), densities.end(), 0.0);
        std::transform(densities.begin(), densities.end(), densities.begin(), [normalization_factor](double density)
                       { return density / normalization_factor; });
        // after transformation, the densities are treated like probabilities
        index_distribution = std::discrete_distribution<size_t>(densities.begin(), densities.end());
    }

    void NaiveCustomCauchy::print(std::ostream &os) const
    {
        os << "NaiveCustomCauchy(gamma="
           << this->gamma
           << ")";
    }

    double NaiveCustomCauchy::draw()
    {
        auto index = index_distribution(rng);
        return standard_support[index];
    }

    std::vector<double> pdf(std::vector<double> zs, const double gamma)
    {
        std::transform(zs.begin(), zs.end(), zs.begin(), [gamma](double z)
                       { return 1 / (1 + std::pow(std::abs(z), gamma)); });
        return zs;
    }

    AdvancedCustomCauchy::AdvancedCustomCauchy(double gamma, const std::mt19937_64 &rng)
    {
        this->gamma = gamma;
        this->rng = rng;
        alpha_gamma = std::gamma_distribution<double>(1.0 - 1.0 / gamma);
        beta_gamma = std::gamma_distribution<double>(1.0 / gamma);
        maybe_negate = std::bernoulli_distribution(0.5);
    }

    void AdvancedCustomCauchy::print(std::ostream &os) const
    {
        os << "AdvancedCustomCauchy(gamma="
           << this->gamma
           << ")";
    }

    double AdvancedCustomCauchy::draw()
    /*
     * This solution is based upon whuber's suggestions made in
     * https://stats.stackexchange.com/a/573470/182357. But that solution would
     * require a beta distribution. Since this kind of distribution is not
     * available in the standard library, I use the link between the gamma
     * distribution and the beta distribution, described here:
     * https://en.wikipedia.org/wiki/Beta_distribution#Derived_from_other_distributions
     *
     */
    {
        auto x = alpha_gamma(rng);
        x = std::max(1e-100, x); // to avoid possible, but very unlikely, division by 0 later
        auto y = alpha_gamma(rng);
        auto z = x / (x + y); // now z is distributed according to the beta distribution
        z = std::pow(1.0 / z - 1.0, 1.0 / gamma);
        if (maybe_negate(rng))
        {
            return -z;
        }
        else
        {
            return z;
        }
    }
}
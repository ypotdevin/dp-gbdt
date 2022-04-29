#include "custom_cauchy.h"

namespace custom_cauchy
{
    CustomStandardCauchy::CustomStandardCauchy(const std::vector<double> standard_support, double gamma, const std::mt19937_64 &rng)
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

    /**
     * @brief Draw a single sample from the standard support (which was provided
     * at initialization of the standard custom Cauchy distribution), according
     * to the pdf.
     */
    double CustomStandardCauchy::draw()
    {
        auto index = index_distribution(rng);
        return standard_support[index];
    }

    /**
     * @brief Return the gamma used to initialize the distribution.
     */
    double CustomStandardCauchy::get_gamma()
    {
        return gamma;
    }

    std::vector<double> pdf(std::vector<double> zs, const double gamma)
    {
        std::transform(zs.begin(), zs.end(), zs.begin(), [gamma](double z)
                       { return 1 / (1 + std::pow(std::abs(z), gamma)); });
        return zs;
    }
}
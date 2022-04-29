#ifndef CUSTOM_CAUCHY_H
#define CUSTOM_CAUCHY_H

#include <algorithm>
#include <random>
#include <vector>

namespace custom_cauchy
{
    class CustomStandardCauchy
    {
    private:
        double gamma;
        std::vector<double> standard_support;
        std::mt19937_64 rng;
        std::discrete_distribution<size_t> index_distribution;

    public:
        CustomStandardCauchy() = default;
        CustomStandardCauchy(const std::vector<double> standard_support, double gamma, const std::mt19937_64 &rng);
        /**
         * @brief Draw a single sample from the standard support (which was provided
         * at initialization of the standard custom Cauchy distribution), according
         * to the pdf.
         */
        double draw();
        /**
         * @brief Return the gamma used to initialize the distribution.
         */
        double get_gamma();

    };

    /**
     * @brief This is the density function z |-> 1 / (1 + |z|^gamma), but
     * operating on a vector (zs) instead of a scalar.
     */
    std::vector<double> pdf(std::vector<double> zs, const double gamma);
}

#endif /* CUSTOM_CAUCHY_H */
#ifndef CUSTOM_CAUCHY_H
#define CUSTOM_CAUCHY_H

#include <algorithm>
#include <random>
#include <vector>

namespace custom_cauchy
{
    class CustomCauchy
    {
    private:
        double gamma;

    public:
        /**
         * @brief Return the gamma used to initialize the distribution.
         */
        double get_gamma() { return gamma; };
        /**
         * @brief Draw a single sample from this distribution.
         */
        virtual double draw() = 0;
        virtual void print(std::ostream &os) const = 0;
        friend std::ostream &operator<<(std::ostream &os, const CustomCauchy &cc)
        {
            cc.print(os);
            return os;
        };
    };

    /**
     * @brief This is the density function z |-> 1 / (1 + |z|^gamma), but
     * operating on a vector (zs) instead of a scalar.
     */
    std::vector<double> pdf(std::vector<double> zs, const double gamma);

    class NaiveCustomCauchy : public CustomCauchy
    {
    private:
        double gamma;
        std::vector<double> standard_support;
        std::mt19937 rng;
        std::discrete_distribution<size_t> index_distribution;

    public:
        NaiveCustomCauchy() = default;
        NaiveCustomCauchy(const std::vector<double> standard_support, double gamma, const std::mt19937 &rng);
        void print(std::ostream &os) const;
        /**
         * @brief Draw a single sample from the standard support (which was provided
         * at initialization of the standard custom Cauchy distribution), according
         * to the pdf.
         */
        double draw();
    };

    /**
     * @brief This variant of the custom distribution does not force us to pick
     * a support set in advance and therefore does not restrict the sample
     * values artificially.
     */
    class AdvancedCustomCauchy : public CustomCauchy
    {
    private:
        double gamma;
        std::mt19937 rng;
        std::gamma_distribution<double> alpha_gamma;
        std::gamma_distribution<double> beta_gamma;
        std::bernoulli_distribution maybe_negate;

    public:
        AdvancedCustomCauchy() = default;
        AdvancedCustomCauchy(double gamma, const std::mt19937 &rng);
        void print(std::ostream &os) const;

        /**
         * @brief Draw a single sample according to this distribution. In
         * contrast to `CustomStandardCauchy::draw()`, there are no artificial
         * boundaries on the drawn values.
         */
        double draw();
    };
}

#endif /* CUSTOM_CAUCHY_H */
#ifndef LAPLACE_H
#define LAPLACE_H

#include <random>

/*
    This method for sampling from laplace distribution is described here:
    https://www.johndcook.com/blog/2018/03/13/generating-laplace-random-variables/
    DPBoost also relies on this mechanism:
    https://github.com/QinbinLi/DPBoost/blob/1174730f9b99aca8389c0721fc3864402236e5cd/include/LightGBM/random_generator.h
*/
class Laplace
{
private:
    std::mt19937 generator;
    std::exponential_distribution<double> distribution;

public:
    Laplace(std::mt19937 rng) : generator(rng){};
    Laplace(int seed) : generator(seed){};

    double return_a_random_variable()
    {
        double e1 = distribution(generator);
        double e2 = distribution(generator);
        return e1 - e2;
    }

    double return_a_random_variable(double _scale)
    {
        return return_a_random_variable() * _scale;
    }
};

#endif /* LAPLACE_H */

#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>
#include <random>
#include "custom_cauchy.h"

// abstract class
class Task
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred) = 0;
    virtual double compute_init_score(std::vector<double> &y) = 0;
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred) = 0;
};

// uses Least Squares as cost/loss function
class Regression : public Task
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);

    // mean
    virtual double compute_init_score(std::vector<double> &y);

    // RMSE
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred);
};

// uses Binomial Deviance as cost/loss function
class BinaryClassification : public Task
{
public:
    // expit
    virtual std::vector<double> compute_gradients(std::vector<double> &y, std::vector<double> &y_pred);

    // logit
    virtual double compute_init_score(std::vector<double> &y);

    // accuracy
    virtual double compute_score(std::vector<double> &y, std::vector<double> &y_pred);
};

/**
 * @param errors The absolute errors (differences) to apply the root mean
 * squared error function on.
 * @param epsilon
 * @param U
 * @param cc
 * @return double
 */
double dp_rms_custom_cauchy(std::vector<double> errors, const double epsilon, const double U, custom_cauchy::CustomCauchy &cc);

/**
 * @param errors The absolute errors (differences) to apply the root mean
 * squared error function on.
 * @param epsilon The privacy budget.
 * @param U The upper bound on the error terms in errors.
 * @param rng The (pseudo) random number generator to use when drawing from the
 * Cauchy distribution.
 * @return double The epsilon-differentially private rMSE estimate of errors.
 */
double dp_rms_cauchy(std::vector<double> errors, const double epsilon, const double U, std::mt19937 &rng);

/**
 * @param errors The precomputed (to avoid having two arguments which then have
 * to be subtracted) absolute errors, sorted ascendingly.
 * @param beta The beta defining the beta-smooth sensitivity.
 * @param U The upper bound on the errors (not squared errors).
 * @return std::tuple<double, double> The beta-smooth sensitivity and the result
 * of the root mean squared error function, i.e. the function
 *
 *     e_1, ..., e_n |-> sqrt((e_1 ** 2 + ... + e_n ** 2) / n).
 */
std::tuple<double, double> rMS_smooth_sensitivity(std::vector<double> errors, const double beta, double U);

/**
 * @brief The local sensitivity of the rMSE function, already operating on the
 * vector of differences (not on two vectors which then will be subtracted).
 *
 * @param x the current squared error to replace.
 * @param substitute the replacement for x.
 * @param s the sum of the squared errors, but *without* x.
 * @param n the number of squared error (terms) in s, plus 1 for x.
 * @param U the upper bound of the squared errors terms in s, and x.
 * @return double The local sensitivity of the root mean squared error function.
 */
double local_sensitivity(const double x, const double substitute, double s, const std::size_t n);

#endif /* LOSS_FUNCTION_H */
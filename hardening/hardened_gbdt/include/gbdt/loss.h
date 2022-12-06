#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>
#include <random>
#include "custom_cauchy.h"

// abstract class
class Task
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y,
                                                  std::vector<double> &y_pred) = 0;
    virtual double compute_init_score(std::vector<double> &y) = 0;
    virtual double compute_score(std::vector<double> &y,
                                 std::vector<double> &y_pred) = 0;
};

// uses Least Squares as cost/loss function
class Regression : public Task
{
public:
    virtual std::vector<double> compute_gradients(std::vector<double> &y,
                                                  std::vector<double> &y_pred);

    // mean
    virtual double compute_init_score(std::vector<double> &y);

    // RMSE
    virtual double compute_score(std::vector<double> &y,
                                 std::vector<double> &y_pred);
};

// uses Binomial Deviance as cost/loss function
class BinaryClassification : public Task
{
public:
    // expit
    virtual std::vector<double> compute_gradients(std::vector<double> &y,
                                                  std::vector<double> &y_pred);

    // logit
    virtual double compute_init_score(std::vector<double> &y);

    // accuracy
    virtual double compute_score(std::vector<double> &y,
                                 std::vector<double> &y_pred);
};

/**
 * @param errors The absolute errors (differences) to apply the root mean
 * squared error function on.
 * @param epsilon
 * @param U
 * @param cc
 * @return double
 */
double dp_rms_custom_cauchy(std::vector<double> errors,
                            const double epsilon,
                            const double U,
                            custom_cauchy::CustomCauchy &cc);

/**
 * @param errors The absolute errors (differences) to apply the root mean
 * squared error function on.
 * @param epsilon The privacy budget.
 * @param U The upper bound on the error terms in errors.
 * @param rng The (pseudo) random number generator to use when drawing from the
 * Cauchy distribution.
 * @return double The epsilon-differentially private rMSE estimate of errors.
 */
double dp_rms_cauchy(std::vector<double> errors,
                     const double epsilon,
                     const double U,
                     std::mt19937 &rng);

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
std::tuple<double, double> rMS_smooth_sensitivity(std::vector<double> errors,
                                                  const double beta,
                                                  double U);

/**
 * @brief Relying on some precomputed values, compute the difference of the root
 * mean (squared) error function values on neighboring error vectors.
 *
 * Assume to have two vectors of errors X and X', both of length n, differing
 * only at position i. Without loss of generality assume, that X[i] = orig_entry
 * and X'[i] = replacement_entry.
 * Further assume that
 *
 *     sum from j=1 to n, j != i over X[j]
 *     == complementary_sum
 *     == sum from j=1 to n, j != i over X'[j].
 *
 * Then
 *
 *     ||rMSE(X) - rMSE(X')||
 *     == fast_rmse_difference(orig_entry,
 *                             replacement_entry,
 *                             complementary_sum,
 *                             n).
 *
 * Note that replacement_entry should be of the same "unit" as the existing
 * entries -- i.e. when the entries are already squared, replacement_entry
 * should be squared too.
 *
 * There is a connection to the local sensitivity of the rMSE function on an
 * error vector X, if there exists an upper bound U on the errors in X, namely:
 *
 *     local_sensitivity(X)
 *     == max(
 *         fast_rmse_difference(max(X), 0.0, s - max(X), n),
 *         fast_rmse_difference(min(X), U, s - min(X), n)
 *     ),
 *
 * where s = sum from j=1 to n over X[j] and n = |X|.
 *
 *
 * @param orig_entry the vector entry subject to change
 * @param replacement_entry orig_entry's replacement
 * @param complementary_sum the sum of the remaining vector elements, excluding
 * orig_entry
 * @param n the length of the vector (including orig_entry)
 * @return the difference of the rMSE function values
 */
double fast_rmse_difference(double orig_entry,
                            double replacement_entry,
                            double complementary_sum,
                            std::size_t n);

#endif /* LOSS_FUNCTION_H */
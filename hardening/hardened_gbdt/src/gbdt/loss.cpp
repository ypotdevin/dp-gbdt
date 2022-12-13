#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <tuple>
#include <exception>
#include <sstream>
#include "constant_time.h"
#include "logging.h"
#include "spdlog/spdlog.h"
#include "loss.h"
#include "utils.h"

namespace
{
    /**
     * @brief
     *
     * @param largest
     * @param smallest
     * @param prefix_sum ATTENTION: updated by callee
     * @param suffix_sum ATTENTION: updated by callee
     * @param n
     * @param U_squared
     * @return double - the local sensitivity at k (k is implicitly given by the
     * state of prefix_sum and suffix_sum.
     */
    double local_sensitivity_at_k(double largest,
                                  double smallest,
                                  double &prefix_sum,
                                  double &suffix_sum,
                                  size_t n,
                                  double U_squared)
    {
        prefix_sum -= largest; // implicitly replaces largest by 0
        suffix_sum -= smallest;
        auto large_zero_diff = fast_rmse_difference(largest, 0, prefix_sum, n);
        auto small_U_diff = fast_rmse_difference(smallest, U_squared, suffix_sum, n);
        auto local_sens = std::max(large_zero_diff, small_U_diff);

        // the worst case neighbor at k + 1 is derived from the worst case
        // neighbor at k by replacing the smallest value by U, see below, and
        // the largest value by 0 (which has already been done further above).
        suffix_sum += U_squared;
        return local_sens;
    }

    /**
     * @brief compare the local sensitivity of the k-th neighbor to the current
     * smooth sensitivity and update if necessary.
     *
     * @param local_sensitivity
     * @param beta
     * @param k
     * @param smooth_sensitivity ATTENTION: updated by callee
     * @param maximizer_k ATTENTION: updated by callee
     * @param maximizer_local_sensitivity ATTENTION: updated by callee
     */
    void compare_local_with_smooth(double local_sensitivity,
                                   double beta,
                                   size_t k,
                                   double &smooth_sensitivity,
                                   size_t &maximizer_k,
                                   double &maximizer_local_sensitivity)
    {
        auto smooth_sens_candidate = local_sensitivity * std::exp(-beta * k);
        if (smooth_sens_candidate > smooth_sensitivity)
        {
            smooth_sensitivity = smooth_sens_candidate;
            maximizer_k = k;
            maximizer_local_sensitivity = local_sensitivity;
        }
    }
}

/* ---------- Regression ---------- */

double Regression::compute_init_score(std::vector<double> &y)
{
    // mean
    double sum = std::accumulate(y.begin(), y.end(), 0.0);
    return sum / y.size();
}

std::vector<double> Regression::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
{
    std::vector<double> gradients(y.size());
    for (size_t i = 0; i < y.size(); i++)
    {
        gradients[i] = y_pred[i] - y[i];
    }
    return gradients;
}

double Regression::compute_score(std::vector<double> &y, std::vector<double> &y_pred)
{
    // RMSE
    std::transform(y.begin(), y.end(),
                   y_pred.begin(), y_pred.begin(), std::minus<double>());
    std::transform(y_pred.begin(), y_pred.end(),
                   y_pred.begin(), [](double &c)
                   { return std::pow(c, 2); });
    double average = std::accumulate(y_pred.begin(), y_pred.end(), 0.0) / y_pred.size();
    double rmse = std::sqrt(average);
    return rmse;
}

/* ---------- Binary Classification ---------- */

double BinaryClassification::compute_init_score(std::vector<double> &y)
{
    // count how many samples are in each of the 2 classes
    double class1 = y[0];
    double counter1 = 0, counter2 = 0;
    for (auto elem : y)
    {
        bool is_class_1 = elem == class1;
        counter1 += is_class_1;
        counter2 += !is_class_1;
    }
    // just need the smaller value
    double smaller_one = constant_time::select(counter1 < counter2, counter1 / y.size(), counter2 / y.size());
    return std::log(smaller_one / (1 - smaller_one));
}

std::vector<double> BinaryClassification::compute_gradients(std::vector<double> &y, std::vector<double> &y_pred)
{
    // positive gradient: expit(y_pred) - y
    // expit(x): (logistic sigmoid function) = 1/(1+exp(-x))
    std::vector<double> gradients(y.size());
    for (size_t i = 0; i < y.size(); i++)
    {
        gradients[i] = 1 / (1 + std::exp(-y_pred[i])) - y[i];
    }
    return gradients;
}

double BinaryClassification::compute_score(std::vector<double> &y, std::vector<double> &y_pred)
{
    // classification task -> transform continuous predictions back to labels
    std::transform(y_pred.begin(), y_pred.end(),
                   y_pred.begin(), [](double &c)
                   { return 1 / (1 + std::exp(-c)); }); // expit
    for (auto &elem : y_pred)
    {
        elem = (elem < 1 - elem) ? 0 : 1;
    }

    // accuracy
    std::vector<bool> correct_preds(y.size());
    for (size_t i = 0; i < y.size(); i++)
    {
        correct_preds[i] = (y[i] == y_pred[i]);
    }
    double true_preds = std::count(correct_preds.begin(), correct_preds.end(), true);
    return true_preds / y.size();
}

double dp_rms_custom_cauchy(std::vector<double> errors, const double epsilon, const double U, custom_cauchy::CustomCauchy &cc)
{
    auto beta = epsilon / (2 * (cc.get_gamma() + 1.0));
    return dp_rms_custom_cauchy(errors, epsilon, beta, U, cc);
}

double dp_rms_custom_cauchy(std::vector<double> errors, double epsilon, double beta, double U, custom_cauchy::CustomCauchy &cc)
{
    std::sort(errors.begin(), errors.end());
    double sens, rmse;
    LOG_INFO("Calculating beta-smooth sensitivity for beta={1}", beta);
    std::tie(sens, rmse) = rMS_smooth_sensitivity(errors, beta, U);
    auto noise = cc.draw();
    auto dp_rmse = rmse + 2 * (cc.get_gamma() + 1) * sens * noise / epsilon;
    return dp_rmse;
}

double dp_rms_cauchy(std::vector<double> errors, const double epsilon, const double U, std::mt19937 &rng)
{
    double beta = epsilon / 6.0;
    return dp_rms_cauchy(errors, epsilon, beta, U, rng);
}

double dp_rms_cauchy(std::vector<double> errors, double epsilon, double beta, double U, std::mt19937 &rng)
{
    std::sort(errors.begin(), errors.end());
    double sens, rmse;
    LOG_INFO("Calculating beta-smooth sensitivity for beta={1}", beta);
    std::tie(sens, rmse) = rMS_smooth_sensitivity(errors, beta, U);
    std::cauchy_distribution<double> distribution(0.0, 1.0);
    auto noise = distribution(rng);
    auto dp_rmse = rmse + 6.0 * sens * noise / epsilon;
    return dp_rmse;
}

std::tuple<double, double> rMS_smooth_sensitivity(std::vector<double> errors, const double beta, double U)
{
    // If U is chosen well, i.e. a true upper bound on the errors, the clipping
    // will have no effect.
    transform(errors.begin(), errors.end(), errors.begin(), [U](double x)
              { x = clamp(x, -U, U); return x * x; });
    auto U_squared = U * U;
    auto sqe_sum = std::accumulate(errors.begin(), errors.end(), 0.0);
    auto n = errors.size();
    auto rmse = std::sqrt(sqe_sum / n);

    auto smooth_sens = -std::numeric_limits<double>::infinity();
    auto prefix_sum = sqe_sum, suffix_sum = sqe_sum;
    // for diagnostics
    double maximizer_local_sens;
    size_t maximizer_k;

    auto local_sens = local_sensitivity_at_k(errors.at(n - 1),
                                             errors.at(0),
                                             prefix_sum,
                                             suffix_sum,
                                             n,
                                             U_squared);
    compare_local_with_smooth(local_sens,
                              beta,
                              0,
                              smooth_sens,
                              maximizer_k,
                              maximizer_local_sens);
    LOG_INFO("### diagnosis value 15 ### - local sensitivity for k = 0: local_sensitivity={1}",
             local_sens);

    for (size_t k = 1; k < n; k++) // traversing the local sensitivities
    {
        auto largest = errors.at(n - k - 1);
        auto smallest = errors.at(k);
        auto current_smooth_sens = local_sensitivity_at_k(largest,
                                                          smallest,
                                                          prefix_sum,
                                                          suffix_sum,
                                                          n,
                                                          U_squared);
        compare_local_with_smooth(current_smooth_sens,
                                  beta,
                                  k,
                                  smooth_sens,
                                  maximizer_k,
                                  maximizer_local_sens);
    }

    auto global_sens = local_sensitivity_at_k(U_squared,
                                              0.0,
                                              prefix_sum,
                                              suffix_sum,
                                              n,
                                              U_squared);
    compare_local_with_smooth(global_sens,
                              beta,
                              n,
                              smooth_sens,
                              maximizer_k,
                              maximizer_local_sens);
    LOG_INFO("### diagnosis value 16 ### - local sensitivity for k = n: global_sensitivity={1}",
             global_sens);

    LOG_INFO("### diagnosis value 04 ### - smooth_sens={1}", smooth_sens);
    LOG_INFO("### diagnosis value 05 ### - maximizer_local_sens={1}", maximizer_local_sens);
    LOG_INFO("### diagnosis value 06 ### - maximizer_k={1}", maximizer_k);
    return std::make_tuple(smooth_sens, rmse);
}

double fast_rmse_difference(double orig_entry,
                            double replacement_entry,
                            double complementary_sum,
                            std::size_t n)
{
    complementary_sum = std::max(complementary_sum, 1e-12); // to avoid division by zero
    auto fast_diff = std::sqrt(complementary_sum / n) *
                     std::abs(std::sqrt(1 + orig_entry / complementary_sum) -
                              std::sqrt(1 + replacement_entry / complementary_sum));
    return fast_diff;
}
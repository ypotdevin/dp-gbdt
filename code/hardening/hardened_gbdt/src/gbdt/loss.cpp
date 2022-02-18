#include "loss.h"
#include <set>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include "constant_time.h"
#include <tuple>
#include <exception>
#include <sstream>

extern bool VERIFICATION_MODE;

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

    if (VERIFICATION_MODE)
    {
        // limit the numbers of decimals to avoid numeric inconsistencies
        std::transform(gradients.begin(), gradients.end(),
                       gradients.begin(), [](double c)
                       { return std::floor(c * 1e15) / 1e15; });
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

    if (VERIFICATION_MODE)
    {
        // limit the numbers of decimals to avoid numeric inconsistencies
        std::transform(gradients.begin(), gradients.end(),
                       gradients.begin(), [](double c)
                       { return std::floor(c * 1e15) / 1e15; });
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

/**
 * @param errors The precomputed errors (to avoid having two arguments which
 * then have to be subtracted.)
 * @param beta The beta defining the beta-smooth sensitivity.
 * @param U The upper bound on the errors (not squared errors).
 * @return std::tuple<double, double> The beta-smooth sensitivity and the result
 * of the root mean squared error function, i.e. the function
 *
 *     e_1, ..., e_n |-> sqrt((e_1 ** 2 + ... + e_n ** 2) / n).
 */
std::tuple<double, double> rMS_smooth_sensitivity(std::vector<double> errors, const double beta, double U)
{
    // TODO: Is this a side channel vulnerability?
    if (U < errors.back())
    {
        std::stringstream message;
        message << "max = " << errors.back() << " is larger than U = " << U << ".\n";
        throw std::invalid_argument(message.str());
    }

    transform(errors.begin(), errors.end(), errors.begin(), [](double x)
              { return x * x; });
    U = U * U;
    auto sqe_sum = std::accumulate(errors.begin(), errors.end(), 0.0);
    auto n = errors.size();
    auto rmse = std::sqrt(sqe_sum / n);

    auto smooth_sens = local_sensitivity(sqe_sum, n, U) * std::exp(1.0); // for k = 0
    auto prefix_sum = sqe_sum, suffix_sum = sqe_sum;
    for (unsigned int k = 1; k <= n; k++) // traversing the local sensitivities
    {
        prefix_sum = prefix_sum - errors.at(n - k);
        suffix_sum = suffix_sum - errors.at(k - 1) + U;
        auto prefix_local_sens = local_sensitivity(prefix_sum, n, U);
        auto suffix_local_sens = local_sensitivity(suffix_sum, n, U);
        auto local_sens = std::max(prefix_local_sens, suffix_local_sens);
        smooth_sens = std::max(smooth_sens, local_sens * std::exp(-beta * k));
    }
    return std::make_tuple(smooth_sens, rmse);
}

/**
 * @param s The sum of the squared errors.
 * @param n The number of squared error (terms) in s.
 * @param U The upper bound of the squared errors terms in s.
 * @return double The local sensitivity of the root mean squared error function.
 */
double local_sensitivity(const double s, const std::size_t n, const double U)
{
    auto ls_a = std::sqrt(U / n);
    auto ls_b = std::sqrt(s / n) * std::abs(std::sqrt(1 + U / s) - 1.0);
    auto ret = constant_time::select(s <= 0.0, ls_a, ls_b);
    return ret;
}

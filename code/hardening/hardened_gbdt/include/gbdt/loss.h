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

// double dp_rms_cauchy(std::vector<double> errors, const double epsilon, const double U);
double dp_rms_custom_cauchy(std::vector<double> errors, const double epsilon, const double U, custom_cauchy::CustomCauchy *cc);
double dp_rms_cauchy(std::vector<double> errors, const double epsilon, const double U, std::mt19937 &rng);
std::tuple<double, double> rMS_smooth_sensitivity(std::vector<double> errors, const double beta, double U);
double local_sensitivity(const double x, const double substitute, double s, const std::size_t n);

#endif /* LOSS_FUNCTION_H */
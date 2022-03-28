#ifndef EVALUATION_H
#define EVALUATION_H

#include "parameters.h"

namespace evaluation
{
    void write_csv_file(const std::string filename, const ModelParams &parameters, const std::string score_metric, const std::vector<double> &train_scores, const std::vector<double> &test_scores);
}

#endif /* EVALUATION_H */
#ifndef PARSER_H
#define PARSER_H

#include "utils.h"
#include "parameters.h"
#include "data.h"
#include "cli_parser.h"

class Parser
{
private:
    // methods
    static std::vector<std::string> split_string(const std::string &s, char delim);
    static DataSet *parse_file(std::string dataset_file, std::string dataset_name, int num_rows, int num_cols, int num_samples,
                               std::shared_ptr<Task> task, std::vector<int> num_idx, std::vector<int> cat_idx, std::vector<int> cat_values, std::vector<int> target_idx,
                               std::vector<int> drop_idx, ModelParams &parameters);

public:
    // methods
    static DataSet *get_abalone(ModelParams &parameters, size_t num_samples);
    static DataSet *get_YearPredictionMSD(ModelParams &parameters,
                                          size_t num_samples);
    static DataSet *get_adult(ModelParams &parameters, size_t num_samples);
    static DataSet *get_concrete(ModelParams &parameters, size_t num_samples);
    static DataSet *get_wine(ModelParams &parameters, size_t num_samples);
};

/**
 * @brief Read from CLI which dataset to use and update model parameters accordingly.
 *
 * Currently parsed parameters:
 *   --dataset (string, like "abalone", "adult", …)
 *   --num-samples (int)
 *
 * @param cp the parser holding the command line arguments.
 * @param mp the model parameters to update.
 * @return DataSet* the dataset identified by argument "--dataset", reading the
 * first "--num-samples" lines.
 */
DataSet *parse_dataset_parameters(cli_parser::CommandLineParser &cp, ModelParams &mp);
DataSet *select_dataset(const std::string &dataset, const size_t num_samples, ModelParams &mp);

#endif // PARSER_H
#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include "dataset_parser.h"
#include "data.h"

/* Parsing:
    - the data file needs to be comma separated
    - so far it only looks out for "?" as missing values, and then gets rid of those rows
    - you need to specify stuff like size, which features are numerical/categorical,
      which feature is the target and which features you want to drop.
    - cat_values allows specifying how many values each categorical feature may have. This allows
      us to hide the fact whether some feature value is actually present in our dataset.

    Given these requirements, it should be easy to add new datasets in the same
    fashion as the ones below. But make sure to double check what you get.
*/

DataSet *Parser::get_abalone(ModelParams &parameters,
                             size_t num_samples)
{
    std::string file = "datasets/real/abalone.data";
    std::string name = "abalone";
    int num_rows = 4177;
    int num_cols = 9;
    std::shared_ptr<Regression> task(new Regression());
    std::vector<int> num_idx = {1, 2, 3, 4, 5, 6, 7};
    std::vector<int> cat_idx = {0};
    std::vector<int> target_idx = {8};
    std::vector<int> drop_idx = {};
    std::vector<int> cat_values = {}; // empty -> will be filled with the present values in the dataset
    parameters.grid_lower_bounds = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    parameters.grid_upper_bounds = {0.0, 1.0, 1.0, 1.5, 3.0, 2.0, 1.0, 1.5};
    parameters.grid_step_sizes = {1.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
                      cat_idx, cat_values, target_idx, drop_idx, parameters);
}

DataSet *Parser::get_YearPredictionMSD(ModelParams &parameters,
                                       size_t num_samples)
{
    std::string file = "datasets/real/YearPredictionMSD.txt";
    std::string name = "yearMSD";
    int num_rows = 515345;
    int num_cols = 91;
    std::shared_ptr<Regression> task(new Regression());
    std::vector<int> num_idx(90);
    std::iota(std::begin(num_idx)++, std::end(num_idx), 1); // num_idx = {1,...,90}
    std::vector<int> cat_idx = {};
    std::vector<int> target_idx = {0};
    std::vector<int> drop_idx = {};
    std::vector<int> cat_values = {}; // empty -> will be filled with the present values in the dataset

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
                      cat_idx, cat_values, target_idx, drop_idx, parameters);
}

DataSet *Parser::get_adult(ModelParams &parameters,
                           size_t num_samples)
{
    std::string file = "datasets/real/adult.data";
    std::string name = "adult";
    int num_rows = 48842;
    int num_cols = 15;
    std::shared_ptr<BinaryClassification> task(new BinaryClassification());
    std::vector<int> num_idx = {0, 4, 10, 11, 12};
    std::vector<int> cat_idx = {1, 3, 5, 6, 7, 8, 9, 13};
    std::vector<int> target_idx = {14};
    std::vector<int> drop_idx = {2};
    std::vector<int> cat_values = {}; // empty -> will be filled with the present values in the dataset

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
                      cat_idx, cat_values, target_idx, drop_idx, parameters);
}

DataSet *Parser::get_concrete(ModelParams &parameters, size_t num_samples)
{
    std::string file = "datasets/real/Concrete_Data_Yeh.csv";
    std::string name = "concrete";
    int num_rows = 1030;
    int num_cols = 9;
    std::shared_ptr<Regression> task(new Regression());
    std::vector<int> num_idx = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int> cat_idx = {};
    std::vector<int> target_idx = {8};
    std::vector<int> drop_idx = {};
    std::vector<int> cat_values = {}; // empty -> will be filled with the present values in the dataset

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
                      cat_idx, cat_values, target_idx, drop_idx, parameters);
}

DataSet *Parser::get_wine(ModelParams &parameters, size_t num_samples)
{
    std::string file = "datasets/real/winequality-red.csv";
    std::string name = "wine";
    int num_rows = 1599;
    int num_cols = 12;
    std::shared_ptr<BinaryClassification> task(new BinaryClassification());
    std::vector<int> num_idx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> cat_idx = {};
    std::vector<int> target_idx = {11};
    std::vector<int> drop_idx = {};
    std::vector<int> cat_values = {};
    parameters.grid_lower_bounds = {4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 6.0, 0.99, 2.5, 0.0, 7.0};
    parameters.grid_upper_bounds = {16.0, 2.0, 1.0, 15.0, 1.0, 80.0, 300.0, 1.0, 5.0, 2, 18};
    parameters.grid_step_sizes = {0.1, 0.01, 0.01, 0.1, 0.001, 1.0, 1.0, 0.0001, 0.01, 0.01, 0.1};

    return parse_file(file, name, num_rows, num_cols, num_samples, task, num_idx,
                      cat_idx, cat_values, target_idx, drop_idx, parameters);
}

/** Utility functions */

std::vector<std::string> Parser::split_string(const std::string &s, char delim)
{
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        result.push_back(item);
    }
    return result;
}

DataSet *Parser::parse_file(std::string dataset_file, std::string dataset_name, int num_rows,
                            int num_cols, int num_samples, std::shared_ptr<Task> task, std::vector<int> num_idx,
                            std::vector<int> cat_idx, std::vector<int> cat_values, std::vector<int> target_idx, std::vector<int> drop_idx,
                            ModelParams &parameters)
{
    std::ifstream infile(dataset_file, std::ifstream::in);
    if (!infile)
    {
        throw std::runtime_error("Could not find dataset file");
    }
    std::string line;
    VVD X;
    std::vector<double> y;
    num_samples = std::min(num_samples, num_rows);

    parameters.num_idx = num_idx;
    parameters.cat_idx = cat_idx;
    parameters.task = task;

    // parse dataset, label-encode categorical features
    int current_index = 0;
    std::vector<std::map<std::string, double>> mappings(num_cols + 1); // last (additional) one is for y

    while (std::getline(infile, line, '\n') && current_index < num_samples)
    {
        std::stringstream ss(line);
        std::vector<std::string> strings = split_string(line, ',');
        std::vector<double> X_row;

        // drop dataset rows that contain missing entries ("?")
        if (line.find('?') < line.length() or line.empty())
        {
            continue;
        }

        // go through each column
        for (size_t i = 0; i < strings.size(); i++)
        {

            // is it a drop column?
            if (std::find(drop_idx.begin(), drop_idx.end(), i) != drop_idx.end())
            {
                continue;
            }

            // y
            if (std::find(target_idx.begin(), target_idx.end(), i) != target_idx.end())
            {
                if (dynamic_cast<Regression *>(task.get()))
                {
                    // regression -> y is numerical
                    y.push_back(stof(strings[i]));
                }
                else
                {
                    try
                    { // categorical
                        double dummy_value = mappings.back().at(strings[i]);
                        y.push_back(dummy_value);
                    }
                    catch (const std::out_of_range &oor)
                    {
                        // new label encountered, create mapping
                        mappings.back().insert({strings[i], mappings.back().size()});
                        double dummy_value = mappings.back().at(strings[i]);
                        y.push_back(dummy_value);
                    }
                }
                continue;
            }

            // X
            if (std::find(num_idx.begin(), num_idx.end(), i) != num_idx.end())
            {
                // numerical feature
                X_row.push_back(stof(strings[i]));
            }
            else
            {
                // categorical feature, do label-encoding
                try
                {
                    double dummy_value = mappings[i].at(strings[i]);
                    X_row.push_back(dummy_value);
                }
                catch (const std::out_of_range &oor)
                {
                    // new label encountered, create mapping
                    mappings[i].insert({strings[i], mappings[i].size()});
                    double dummy_value = mappings[i].at(strings[i]);
                    X_row.push_back(dummy_value);
                }
            }
        }
        X.push_back(X_row);
        current_index++;
    }

    DataSet *dataset = new DataSet(X, y);
    dataset->name = std::string(dataset_name) + std::string("_size_") + std::to_string(num_samples);

    // adjust cat_values if necessary
    if (cat_values.empty())
    {
        for (size_t i = 0; i < mappings.size() - 1; i++)
        {
            auto map = mappings[i];
            std::vector<double> keys;
            if (std::find(cat_idx.begin(), cat_idx.end(), i) == cat_idx.end())
            {
                parameters.cat_values.push_back(keys);
                continue;
            }
            for (auto it = map.begin(); it != map.end(); it++)
            {
                keys.push_back(it->second);
            }
            parameters.cat_values.push_back(keys);
        }
    }
    else
    {
        for (int i = 0; i < num_cols; i++)
        {
            std::vector<double> keys;
            if (std::find(cat_idx.begin(), cat_idx.end(), i) == cat_idx.end())
            {
                parameters.cat_values.push_back(keys);
                continue;
            }
            for (double j = 0.0; j < cat_values[i]; j++)
            {
                keys.push_back(j);
            }
            parameters.cat_values.push_back(keys);
        }
    }

    return dataset;
}

/**
 * @brief Based on a label, parse one of the predefined datasets.
 *
 * @param dataset the label of the dataset (e.g. "abalone", "adult", …)
 * @param num_samples the number of samples to draw from the dataset.
 * @param mp the model parameters. Its dataset specific subset of parameters
 * will be manipulated by this.
 * @return DataSet* the dataset matching label
 */
DataSet *select_dataset(const std::string &dataset, const size_t num_samples, ModelParams &mp)
{
    if (dataset == "abalone")
    {
        return Parser::get_abalone(mp, num_samples);
    }
    else if (dataset == "adult")
    {
        return Parser::get_adult(mp, num_samples);
    }
    else if (dataset == "YearPredictionMSD")
    {
        return Parser::get_YearPredictionMSD(mp, num_samples);
    }
    else if (dataset == "concrete")
    {
        return Parser::get_concrete(mp, num_samples);
    }
    else if (dataset == "wine")
    {
        return Parser::get_wine(mp, num_samples);
    }
    else
    {
        std::string msg = "Unknown dataset: ";
        msg.append(dataset);
        msg.append(".\n");
        throw std::runtime_error(msg);
    }
}
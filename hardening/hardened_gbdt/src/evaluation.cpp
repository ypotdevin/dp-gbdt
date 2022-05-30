#include "evaluation.h"
#include "csvfile.h"
#include "utils.h"

namespace evaluation
{
    /**
     * @brief Write training parameters, train scores and test scores (of the
     * cross validation) to a .csv file.
     *
     * @param filename the name of the resulting .csv file.
     * @param parameters the model parameters.
     * @param score_metric the metric used to calculate the score.
     * @param train_scores the scores on the training set splits.
     * @param test_scores the scores on the test set splits.
     * @author Yannik Potdevin
     */
    void write_csv_file(const std::string filename, const ModelParams &parameters, const std::string score_metric, const std::vector<double> &train_scores, const std::vector<double> &test_scores, const int seed)
    {
        try
        {
            csvfile csv(filename); // may throw an exception

            std::vector<std::string> header = {
                "param_gradient_filtering",
                "param_l2_lambda",
                "param_l2_threshold",
                "param_learning_rate",
                "param_max_depth",
                "param_nb_trees",
                "param_ensemble_privacy_budget",
                "param_tree_rejector",
                "seed",
                "score_metric"};
            for (std::size_t i = 0; i < train_scores.size(); i++)
            {
                std::ostringstream col_name;
                col_name << "split_" << i << "_train_score";
                header.push_back(col_name.str());
            }
            for (std::size_t i = 0; i < test_scores.size(); i++)
            {
                std::ostringstream col_name;
                col_name << "split_" << i << "_test_score";
                header.push_back(col_name.str());
            }
            csv.write_header(header);

            // pay attention to match the sequence of the header
            csv << parameters.gradient_filtering
                << parameters.l2_lambda
                << parameters.l2_threshold
                << parameters.learning_rate
                << parameters.max_depth
                << parameters.nb_trees
                << parameters.privacy_budget
                << *parameters.tree_rejector
                << seed
                << score_metric;
            for (auto train_score : train_scores)
            {
                csv << train_score;
            }
            for (auto test_score : test_scores)
            {
                csv << test_score;
            }

            csv << endrow;
        }
        catch (const std::exception &ex)
        {
            std::cout << "Unable to create .csv file: " << ex.what() << std::endl;
        }
    }
}
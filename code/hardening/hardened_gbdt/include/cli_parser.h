#ifndef CLI_PARSER_H
#define CLI_PARSER_H

#include <algorithm>
#include <string>
#include <vector>

namespace cli_parser
{
    /**
     * This class is heavily inspired by iain's (https://stackoverflow.com/users/85381/iain)
     * code here: https://stackoverflow.com/a/868894.
     *
     * @brief Parse simple command line arguments (as strings).
     */
    class CommandLineParser
    {
    public:
        CommandLineParser(const int argc, const char *const *argv)
        {
            for (int i = 1; i < argc; ++i)
                this->arguments.push_back(std::string(argv[i]));
        }

        bool hasOption(const std::string &option) const
        {
            return std::find(this->arguments.begin(), this->arguments.end(), option) != this->arguments.end();
        }

        const std::string &getOptionValue(const std::string &option) const
        {
            std::vector<std::string>::const_iterator option_value;
            option_value = std::find(this->arguments.begin(), this->arguments.end(), option);
            if (option_value != this->arguments.end() && ++option_value != this->arguments.end())
            {
                return *option_value;
            }
            static const std::string empty_std_string("");
            return empty_std_string;
        }

        double getDoubleOptionValue(const std::string &option)
        {
            double value = std::stod(getOptionValue(option));
            return value;
        }

        int getIntOptionValue(const std::string &option)
        {
            int value = std::stoi(getOptionValue(option));
            return value;
        }

    private:
        std::vector<std::string> arguments;
    };
}

#endif /* CLI_PARSER_H */

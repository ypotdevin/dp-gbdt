#include <algorithm>
#include "cli_parser.h"

namespace cli_parser
{
    CommandLineParser::CommandLineParser(const int argc, const char *const *argv)
    {
        for (int i = 1; i < argc; ++i)
            this->arguments.push_back(std::string(argv[i]));
    }

    bool CommandLineParser::hasOption(const std::string &option) const
    {
        return std::find(this->arguments.begin(), this->arguments.end(), option) != this->arguments.end();
    }

    const std::string &CommandLineParser::getOptionValue(const std::string &option) const
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

    double CommandLineParser::getDoubleOptionValue(const std::string &option)
    {
        double value = std::stod(getOptionValue(option));
        return value;
    }

    int CommandLineParser::getIntOptionValue(const std::string &option)
    {
        int value = std::stoi(getOptionValue(option));
        return value;
    }
}

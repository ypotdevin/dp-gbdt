#ifndef CLI_PARSER_H
#define CLI_PARSER_H

#include <string>
#include <vector>

namespace cli_parser
{
    /**
     * This class is heavily inspired by iain's (https://stackoverflow.com/users/85381/iain)
     * code here: https://stackoverflow.com/a/868894.
     *
     * @brief Parse simple command line arguments.
     */
    class CommandLineParser
    {
    public:
        CommandLineParser(const int argc, const char *const *argv);
        bool hasOption(const std::string &option) const;
        const std::string &getOptionValue(const std::string &option) const;
        double getDoubleOptionValue(const std::string &option);
        int getIntOptionValue(const std::string &option);

    private:
        std::vector<std::string> arguments;
    };
}

#endif /* CLI_PARSER_H */

// Copyright 2022 Vladimir Shestakov (https://github.com/rudolfovich)

// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Source: https://gist.github.com/rudolfovich/f250900f1a833e715260a66c87369d15
#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

class csvfile;

inline static csvfile &endrow(csvfile &file);
inline static csvfile &flush(csvfile &file);

class csvfile
{
    std::ofstream fs_;
    bool is_first_;
    const std::string separator_;
    const std::string escape_seq_;
    const std::string special_chars_;

public:
    csvfile(const std::string filename, const std::string separator = ";")
        : fs_(), is_first_(true), separator_(separator), escape_seq_("\""), special_chars_("\"")
    {
        fs_.exceptions(std::ios::failbit | std::ios::badbit);
        fs_.open(filename);
    }

    ~csvfile()
    {
        flush();
        fs_.close();
    }

    void flush()
    {
        fs_.flush();
    }

    void endrow()
    {
        fs_ << std::endl;
        is_first_ = true;
    }

    csvfile &operator<<(csvfile &(*val)(csvfile &))
    {
        return val(*this);
    }

    csvfile &operator<<(const char *val)
    {
        return write(escape(val));
    }

    csvfile &operator<<(const std::string &val)
    {
        return write(escape(val));
    }

    template <typename T>
    csvfile &operator<<(const T &val)
    {
        return write(val);
    }

    /**
     * @brief Given a vector of column names, write them to the csv file.
     * To ensure the result becomes in fact a header, call this method before
     * all others after object creation.
     *
     * @param header the vector of column names.
     * @author Yannik Potdevin
     */
    void write_header(std::vector<std::string> header)
    {
        for (auto col_name : header)
        {
            write(col_name);
        }
        endrow();
    }

private:
    template <typename T>
    csvfile &write(const T &val)
    {
        if (!is_first_)
        {
            fs_ << separator_;
        }
        else
        {
            is_first_ = false;
        }
        fs_ << val;
        return *this;
    }

    std::string escape(const std::string &val)
    {
        std::ostringstream result;
        result << '"';
        std::string::size_type to, from = 0u, len = val.length();
        while (from < len &&
               std::string::npos != (to = val.find_first_of(special_chars_, from)))
        {
            result << val.substr(from, to - from) << escape_seq_ << val[to];
            from = to + 1;
        }
        result << val.substr(from) << '"';
        return result.str();
    }
};

inline static csvfile &endrow(csvfile &file)
{
    file.endrow();
    return file;
}

inline static csvfile &flush(csvfile &file)
{
    file.flush();
    return file;
}
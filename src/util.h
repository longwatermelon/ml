#pragma once
#include "emel/util.h"
#include <sstream>

// read contents from file into string
inline string read_file(const string &path) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

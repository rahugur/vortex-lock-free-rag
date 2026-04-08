#pragma once
// JSON alias — same convention as Forge for seamless interop.

#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <string>

namespace vortex {

using Json = nlohmann::json;

namespace json_util {

inline Json parse_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    Json j;
    try {
        ifs >> j;
    } catch (const Json::parse_error& e) {
        throw std::runtime_error("JSON parse error in " + path + ": " + e.what());
    }
    return j;
}

}  // namespace json_util
}  // namespace vortex

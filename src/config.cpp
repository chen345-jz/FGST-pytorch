#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace mmwave {

namespace {
std::string trim(const std::string& s) {
    auto begin = s.begin();
    while (begin != s.end() && std::isspace(static_cast<unsigned char>(*begin))) {
        ++begin;
    }
    auto end = s.end();
    while (end != begin && std::isspace(static_cast<unsigned char>(*(end - 1)))) {
        --end;
    }
    return std::string(begin, end);
}
}  // namespace

bool Config::load(const std::string& path) {
    kv_.clear();
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    std::string line;
    while (std::getline(ifs, line)) {
        const auto clean = trim(line);
        if (clean.empty() || clean[0] == '#') continue;
        const auto pos = clean.find('=');
        if (pos == std::string::npos) continue;
        const auto key = trim(clean.substr(0, pos));
        const auto val = trim(clean.substr(pos + 1));
        if (!key.empty()) kv_[key] = val;
    }
    return true;
}

std::string Config::get(const std::string& key, const std::string& default_value) const {
    const auto it = kv_.find(key);
    return it == kv_.end() ? default_value : it->second;
}

int Config::get_int(const std::string& key, int default_value) const {
    const auto v = get(key, "");
    if (v.empty()) return default_value;
    try {
        return std::stoi(v);
    } catch (...) {
        return default_value;
    }
}

float Config::get_float(const std::string& key, float default_value) const {
    const auto v = get(key, "");
    if (v.empty()) return default_value;
    try {
        return std::stof(v);
    } catch (...) {
        return default_value;
    }
}

bool Config::get_bool(const std::string& key, bool default_value) const {
    const auto v = get(key, "");
    if (v.empty()) return default_value;
    const std::string lower = [&]() {
        std::string s = v;
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    }();
    if (lower == "1" || lower == "true" || lower == "yes") return true;
    if (lower == "0" || lower == "false" || lower == "no") return false;
    return default_value;
}

}  // namespace mmwave

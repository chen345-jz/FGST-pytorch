#pragma once

#include <string>
#include <unordered_map>

namespace mmwave {

class Config {
public:
    bool load(const std::string& path);
    std::string get(const std::string& key, const std::string& default_value = "") const;
    int get_int(const std::string& key, int default_value) const;
    float get_float(const std::string& key, float default_value) const;
    bool get_bool(const std::string& key, bool default_value) const;

private:
    std::unordered_map<std::string, std::string> kv_;
};

}  // namespace mmwave

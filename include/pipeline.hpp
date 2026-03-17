#pragma once

#include <string>

namespace mmwave {

int run_train(const std::string& config_path);
int run_eval(const std::string& config_path);
int run_predict(const std::string& config_path, const std::string& sample_csv_path);

}  // namespace mmwave

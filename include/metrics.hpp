#pragma once

#include <string>
#include <vector>

namespace mmwave {

struct MetricsReport {
    float accuracy = 0.0f;
    std::vector<int> labels;
    std::vector<std::vector<int>> confusion;
    std::vector<float> precision;
    std::vector<float> recall;
    std::vector<float> f1;
    float macro_f1 = 0.0f;
};

MetricsReport compute_metrics(const std::vector<int>& y_true, const std::vector<int>& y_pred);
bool save_metrics_csv(const std::string& path, const MetricsReport& report);
void print_metrics(const MetricsReport& report);

}  // namespace mmwave

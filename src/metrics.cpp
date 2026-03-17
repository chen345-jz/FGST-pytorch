#include "metrics.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace mmwave {

MetricsReport compute_metrics(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    MetricsReport r;
    if (y_true.empty() || y_true.size() != y_pred.size()) return r;

    std::vector<int> labels = y_true;
    labels.insert(labels.end(), y_pred.begin(), y_pred.end());
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    r.labels = labels;

    std::unordered_map<int, int> idx;
    for (std::size_t i = 0; i < labels.size(); ++i) idx[labels[i]] = static_cast<int>(i);

    const int n = static_cast<int>(labels.size());
    r.confusion.assign(n, std::vector<int>(n, 0));
    int correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        int a = idx[y_true[i]];
        int b = idx[y_pred[i]];
        r.confusion[a][b]++;
        if (a == b) correct++;
    }
    r.accuracy = static_cast<float>(correct) / static_cast<float>(y_true.size());

    r.precision.assign(n, 0.0f);
    r.recall.assign(n, 0.0f);
    r.f1.assign(n, 0.0f);
    float f1_sum = 0.0f;
    for (int c = 0; c < n; ++c) {
        int tp = r.confusion[c][c];
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < n; ++i) {
            if (i != c) {
                fp += r.confusion[i][c];
                fn += r.confusion[c][i];
            }
        }
        const float p = (tp + fp) ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
        const float rec = (tp + fn) ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
        const float f = (p + rec) > 0.0f ? 2.0f * p * rec / (p + rec) : 0.0f;
        r.precision[c] = p;
        r.recall[c] = rec;
        r.f1[c] = f;
        f1_sum += f;
    }
    r.macro_f1 = n > 0 ? f1_sum / static_cast<float>(n) : 0.0f;
    return r;
}

bool save_metrics_csv(const std::string& path, const MetricsReport& r) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;
    ofs << "accuracy," << r.accuracy << "\n";
    ofs << "macro_f1," << r.macro_f1 << "\n";
    ofs << "label,precision,recall,f1\n";
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
        ofs << r.labels[i] << "," << r.precision[i] << "," << r.recall[i] << "," << r.f1[i] << "\n";
    }
    ofs << "confusion_matrix\n";
    ofs << "actual/pred";
    for (int lab : r.labels) ofs << "," << lab;
    ofs << "\n";
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
        ofs << r.labels[i];
        for (std::size_t j = 0; j < r.labels.size(); ++j) ofs << "," << r.confusion[i][j];
        ofs << "\n";
    }
    return ofs.good();
}

void print_metrics(const MetricsReport& r) {
    std::cout << "accuracy: " << r.accuracy << "\n";
    std::cout << "macro_f1: " << r.macro_f1 << "\n";
    for (std::size_t i = 0; i < r.labels.size(); ++i) {
        std::cout << "label " << r.labels[i] << " -> P: " << r.precision[i] << ", R: " << r.recall[i]
                  << ", F1: " << r.f1[i] << "\n";
    }
}

}  // namespace mmwave

#include "classifier.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mmwave {

void KNNClassifier::fit(const FeatureMatrix& x, const std::vector<int>& y) {
    train_x_ = x;
    train_y_ = y;
}

float KNNClassifier::l2_distance_sq(const FeatureVector& a, const FeatureVector& b) const {
    const std::size_t n = std::min(a.size(), b.size());
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

int KNNClassifier::predict_one(const FeatureVector& sample) const {
    if (train_x_.empty()) return -1;
    std::vector<std::pair<float, int>> dist_label;
    dist_label.reserve(train_x_.size());
    for (std::size_t i = 0; i < train_x_.size(); ++i) {
        dist_label.push_back({l2_distance_sq(sample, train_x_[i]), train_y_[i]});
    }
    std::sort(dist_label.begin(), dist_label.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    const int k = std::max(1, std::min(cfg_.k_neighbors, static_cast<int>(dist_label.size())));
    std::unordered_map<int, int> vote;
    for (int i = 0; i < k; ++i) vote[dist_label[i].second]++;

    int best_label = -1;
    int best_count = -1;
    for (const auto& kv : vote) {
        if (kv.second > best_count) {
            best_count = kv.second;
            best_label = kv.first;
        }
    }
    return best_label;
}

std::vector<int> KNNClassifier::predict(const FeatureMatrix& x) const {
    std::vector<int> out;
    out.reserve(x.size());
    for (const auto& s : x) out.push_back(predict_one(s));
    return out;
}

float KNNClassifier::accuracy(const FeatureMatrix& x, const std::vector<int>& y_true) const {
    if (x.empty() || y_true.empty() || x.size() != y_true.size()) return 0.0f;
    const auto pred = predict(x);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        if (pred[i] == y_true[i]) ++correct;
    }
    return static_cast<float>(correct) / static_cast<float>(pred.size());
}

}  // namespace mmwave

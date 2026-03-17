#include "softmax_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace mmwave {

namespace {
float safe_exp(float x) {
    if (x > 40.0f) x = 40.0f;
    if (x < -40.0f) x = -40.0f;
    return std::exp(x);
}
}  // namespace

std::vector<float> SoftmaxClassifier::logits(const FeatureVector& x) const {
    std::vector<float> out(num_classes_, 0.0f);
    if (num_classes_ == 0 || num_features_ == 0) return out;
    for (int c = 0; c < num_classes_; ++c) {
        const int base = c * (num_features_ + 1);
        float z = weights_[base + num_features_];
        for (int f = 0; f < num_features_ && f < static_cast<int>(x.size()); ++f) {
            z += weights_[base + f] * x[f];
        }
        out[c] = z;
    }
    return out;
}

int SoftmaxClassifier::argmax(const std::vector<float>& v) const {
    int idx = 0;
    for (int i = 1; i < static_cast<int>(v.size()); ++i) {
        if (v[i] > v[idx]) idx = i;
    }
    return idx;
}

void SoftmaxClassifier::fit(const FeatureMatrix& x, const std::vector<int>& y) {
    if (x.empty() || y.empty() || x.size() != y.size()) return;
    num_features_ = static_cast<int>(x[0].size());

    labels_ = y;
    std::sort(labels_.begin(), labels_.end());
    labels_.erase(std::unique(labels_.begin(), labels_.end()), labels_.end());
    num_classes_ = static_cast<int>(labels_.size());
    if (num_classes_ <= 1) return;

    std::unordered_map<int, int> lab2idx;
    for (int i = 0; i < num_classes_; ++i) lab2idx[labels_[i]] = i;

    weights_.assign(static_cast<std::size_t>(num_classes_ * (num_features_ + 1)), 0.0f);

    for (int ep = 0; ep < cfg_.epochs; ++ep) {
        for (std::size_t i = 0; i < x.size(); ++i) {
            const auto z = logits(x[i]);
            float max_z = z[0];
            for (float v : z) max_z = std::max(max_z, v);
            std::vector<float> p(num_classes_, 0.0f);
            float s = 0.0f;
            for (int c = 0; c < num_classes_; ++c) {
                p[c] = safe_exp(z[c] - max_z);
                s += p[c];
            }
            for (int c = 0; c < num_classes_; ++c) p[c] /= s;

            const int yi = lab2idx[y[i]];
            for (int c = 0; c < num_classes_; ++c) {
                const float grad_common = p[c] - (c == yi ? 1.0f : 0.0f);
                const int base = c * (num_features_ + 1);
                for (int f = 0; f < num_features_; ++f) {
                    const float grad = grad_common * x[i][f] + cfg_.l2 * weights_[base + f];
                    weights_[base + f] -= cfg_.learning_rate * grad;
                }
                weights_[base + num_features_] -= cfg_.learning_rate * grad_common;
            }
        }
    }
}

int SoftmaxClassifier::predict_one(const FeatureVector& sample) const {
    if (num_classes_ == 0 || labels_.empty()) return -1;
    const auto z = logits(sample);
    const int idx = argmax(z);
    return labels_[idx];
}

std::vector<int> SoftmaxClassifier::predict(const FeatureMatrix& x) const {
    std::vector<int> out;
    out.reserve(x.size());
    for (const auto& s : x) out.push_back(predict_one(s));
    return out;
}

float SoftmaxClassifier::accuracy(const FeatureMatrix& x, const std::vector<int>& y_true) const {
    if (x.empty() || y_true.empty() || x.size() != y_true.size()) return 0.0f;
    const auto p = predict(x);
    int correct = 0;
    for (std::size_t i = 0; i < p.size(); ++i) if (p[i] == y_true[i]) correct++;
    return static_cast<float>(correct) / static_cast<float>(p.size());
}

void SoftmaxClassifier::load_state(std::vector<int> labels, std::vector<float> weights, int num_classes,
                                   int num_features, SoftmaxConfig cfg) {
    labels_ = std::move(labels);
    weights_ = std::move(weights);
    num_classes_ = num_classes;
    num_features_ = num_features;
    cfg_ = cfg;
}

}  // namespace mmwave

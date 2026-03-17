#pragma once

#include <vector>

#include "types.hpp"

namespace mmwave {

struct SoftmaxConfig {
    float learning_rate = 0.05f;
    int epochs = 200;
    float l2 = 1e-4f;
};

class SoftmaxClassifier {
public:
    explicit SoftmaxClassifier(SoftmaxConfig cfg) : cfg_(cfg) {}

    void fit(const FeatureMatrix& x, const std::vector<int>& y);
    int predict_one(const FeatureVector& sample) const;
    std::vector<int> predict(const FeatureMatrix& x) const;
    float accuracy(const FeatureMatrix& x, const std::vector<int>& y_true) const;

    const std::vector<int>& labels() const { return labels_; }
    const std::vector<float>& weights() const { return weights_; }
    int num_classes() const { return num_classes_; }
    int num_features() const { return num_features_; }
    const SoftmaxConfig& config() const { return cfg_; }

    void load_state(std::vector<int> labels, std::vector<float> weights, int num_classes, int num_features,
                    SoftmaxConfig cfg);

private:
    std::vector<float> logits(const FeatureVector& x) const;
    int argmax(const std::vector<float>& v) const;

    SoftmaxConfig cfg_;
    std::vector<int> labels_;
    std::vector<float> weights_;  // [num_classes x (num_features + 1)], last is bias
    int num_classes_ = 0;
    int num_features_ = 0;
};

}  // namespace mmwave

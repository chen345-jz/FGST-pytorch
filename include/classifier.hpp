#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "types.hpp"

namespace mmwave {

struct ClassifierConfig {
    int k_neighbors = 3;
};

class KNNClassifier {
public:
    explicit KNNClassifier(ClassifierConfig cfg) : cfg_(cfg) {}

    void fit(const FeatureMatrix& x, const std::vector<int>& y);
    int predict_one(const FeatureVector& sample) const;
    std::vector<int> predict(const FeatureMatrix& x) const;
    float accuracy(const FeatureMatrix& x, const std::vector<int>& y_true) const;

    const FeatureMatrix& train_x() const { return train_x_; }
    const std::vector<int>& train_y() const { return train_y_; }
    const ClassifierConfig& config() const { return cfg_; }

private:
    float l2_distance_sq(const FeatureVector& a, const FeatureVector& b) const;

    ClassifierConfig cfg_;
    FeatureMatrix train_x_;
    std::vector<int> train_y_;
};

}  // namespace mmwave

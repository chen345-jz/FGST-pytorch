#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace mmwave {

struct FGSTConfig {
    int epochs = 50;
    int batch_size = 16;
    float learning_rate = 1e-3f;
    float weight_decay = 1e-4f;
    int max_frames = 32;
    int max_points_per_frame = 64;
    int num_body_parts = 4;
    int point_feature_dim = 64;
    int temporal_feature_dim = 128;
};

class FGSTModel {
public:
    explicit FGSTModel(FGSTConfig cfg) : cfg_(cfg) {}
    bool available() const;
    bool fit(const std::vector<Sequence>& train_data, const std::vector<int>& train_labels);
    int predict_one(const Sequence& sample) const;
    std::vector<int> predict(const std::vector<Sequence>& data) const;
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    std::vector<int> labels_;
    std::string last_saved_path_;
    FGSTConfig cfg_;
};

}  // namespace mmwave

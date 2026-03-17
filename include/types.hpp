#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mmwave {

struct Point {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float doppler = 0.0f;
    float snr = 0.0f;
};

struct Frame {
    std::vector<Point> points;
    std::int64_t timestamp = 0;
};

struct Sequence {
    std::string id;
    int label = -1;
    std::vector<Frame> frames;
};

struct Dataset {
    std::vector<Sequence> samples;
    std::unordered_map<int, std::string> label_names;
};

struct SplitData {
    std::vector<Sequence> train;
    std::vector<Sequence> test;
};

using FeatureVector = std::vector<float>;
using FeatureMatrix = std::vector<FeatureVector>;

}  // namespace mmwave

#pragma once

#include <vector>

#include "types.hpp"

namespace mmwave {

struct PreprocessParams {
    float min_snr = 0.0f;
    float min_range = 0.0f;
    float max_range = 100.0f;
    std::size_t min_points_per_frame = 1;
};

struct Cluster {
    std::vector<Point> points;
    Point centroid;
};

class Preprocessor {
public:
    explicit Preprocessor(PreprocessParams params) : params_(params) {}
    void filter_sequence(Sequence& sequence) const;
    std::vector<Cluster> cluster_frame_dbscan(const Frame& frame, float eps, std::size_t min_pts) const;

private:
    PreprocessParams params_;
};

}  // namespace mmwave

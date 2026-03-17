#include "features.hpp"

#include <algorithm>
#include <cmath>

namespace mmwave {

namespace {
float sq(float v) { return v * v; }
}

FeatureVector FeatureExtractor::extract(const Sequence& sequence, const Preprocessor& preprocessor,
                                        float cluster_eps, std::size_t cluster_min_pts) const {
    // Feature set:
    // [0] avg points/frame
    // [1] avg cluster count/frame
    // [2] centroid speed mean
    // [3] centroid speed std
    // [4] doppler mean
    // [5] doppler std
    // [6] bbox volume mean
    // [7] bbox volume std
    FeatureVector feat(8, 0.0f);
    if (sequence.frames.empty()) return feat;

    std::vector<float> points_per_frame;
    std::vector<float> cluster_per_frame;
    std::vector<float> speeds;
    std::vector<float> dopplers;
    std::vector<float> bbox_vols;

    points_per_frame.reserve(sequence.frames.size());
    cluster_per_frame.reserve(sequence.frames.size());

    Point last_ctr{};
    bool has_last = false;
    std::int64_t last_ts = 0;
    for (const auto& f : sequence.frames) {
        points_per_frame.push_back(static_cast<float>(f.points.size()));
        if (f.points.empty()) continue;

        auto clusters = preprocessor.cluster_frame_dbscan(f, cluster_eps, cluster_min_pts);
        cluster_per_frame.push_back(static_cast<float>(clusters.size()));

        Point frame_ctr{};
        float min_x = f.points[0].x, max_x = f.points[0].x;
        float min_y = f.points[0].y, max_y = f.points[0].y;
        float min_z = f.points[0].z, max_z = f.points[0].z;
        for (const auto& p : f.points) {
            frame_ctr.x += p.x;
            frame_ctr.y += p.y;
            frame_ctr.z += p.z;
            frame_ctr.doppler += p.doppler;
            dopplers.push_back(p.doppler);
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
            min_z = std::min(min_z, p.z);
            max_z = std::max(max_z, p.z);
        }
        const float inv = 1.0f / static_cast<float>(f.points.size());
        frame_ctr.x *= inv;
        frame_ctr.y *= inv;
        frame_ctr.z *= inv;
        frame_ctr.doppler *= inv;

        const float vol = std::max(0.0f, (max_x - min_x) * (max_y - min_y) * (max_z - min_z));
        bbox_vols.push_back(vol);

        if (has_last) {
            const float dt = std::max(1.0f, static_cast<float>(f.timestamp - last_ts)) * 1e-3f;
            const float d =
                std::sqrt(sq(frame_ctr.x - last_ctr.x) + sq(frame_ctr.y - last_ctr.y) + sq(frame_ctr.z - last_ctr.z));
            speeds.push_back(d / dt);
        }
        last_ctr = frame_ctr;
        last_ts = f.timestamp;
        has_last = true;
    }

    auto mean = [](const std::vector<float>& v) {
        if (v.empty()) return 0.0f;
        float s = 0.0f;
        for (float x : v) s += x;
        return s / static_cast<float>(v.size());
    };
    auto stdev = [&](const std::vector<float>& v) {
        if (v.size() < 2) return 0.0f;
        const float m = mean(v);
        float s2 = 0.0f;
        for (float x : v) s2 += sq(x - m);
        return std::sqrt(s2 / static_cast<float>(v.size() - 1));
    };

    feat[0] = mean(points_per_frame);
    feat[1] = mean(cluster_per_frame);
    feat[2] = mean(speeds);
    feat[3] = stdev(speeds);
    feat[4] = mean(dopplers);
    feat[5] = stdev(dopplers);
    feat[6] = mean(bbox_vols);
    feat[7] = stdev(bbox_vols);

    return feat;
}

}  // namespace mmwave

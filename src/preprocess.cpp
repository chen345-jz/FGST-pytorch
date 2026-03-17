#include "preprocess.hpp"

#include <cmath>
#include <queue>

namespace mmwave {

namespace {
float range_of(const Point& p) { return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z); }

float dist_sq(const Point& a, const Point& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}
}  // namespace

void Preprocessor::filter_sequence(Sequence& sequence) const {
    std::vector<Frame> kept;
    kept.reserve(sequence.frames.size());
    for (auto& f : sequence.frames) {
        Frame out;
        out.timestamp = f.timestamp;
        out.points.reserve(f.points.size());
        for (const auto& p : f.points) {
            const float r = range_of(p);
            if (p.snr < params_.min_snr) continue;
            if (r < params_.min_range || r > params_.max_range) continue;
            out.points.push_back(p);
        }
        if (out.points.size() >= params_.min_points_per_frame) kept.push_back(std::move(out));
    }
    sequence.frames = std::move(kept);
}

std::vector<Cluster> Preprocessor::cluster_frame_dbscan(const Frame& frame, float eps,
                                                        std::size_t min_pts) const {
    std::vector<Cluster> clusters;
    if (frame.points.empty()) return clusters;
    const float eps_sq = eps * eps;
    const int n = static_cast<int>(frame.points.size());
    std::vector<int> labels(n, -2);  // -2 unvisited, -1 noise, >=0 cluster id

    auto region_query = [&](int idx) {
        std::vector<int> neigh;
        for (int j = 0; j < n; ++j) {
            if (dist_sq(frame.points[idx], frame.points[j]) <= eps_sq) neigh.push_back(j);
        }
        return neigh;
    };

    int cluster_id = 0;
    for (int i = 0; i < n; ++i) {
        if (labels[i] != -2) continue;
        auto neighbors = region_query(i);
        if (neighbors.size() < min_pts) {
            labels[i] = -1;
            continue;
        }

        labels[i] = cluster_id;
        std::queue<int> q;
        for (int nb : neighbors) q.push(nb);

        while (!q.empty()) {
            const int cur = q.front();
            q.pop();
            if (labels[cur] == -1) labels[cur] = cluster_id;
            if (labels[cur] != -2) continue;
            labels[cur] = cluster_id;
            auto cur_neighbors = region_query(cur);
            if (cur_neighbors.size() >= min_pts) {
                for (int nb2 : cur_neighbors) q.push(nb2);
            }
        }
        ++cluster_id;
    }

    clusters.resize(cluster_id);
    for (int i = 0; i < n; ++i) {
        const int id = labels[i];
        if (id >= 0) clusters[id].points.push_back(frame.points[i]);
    }

    for (auto& c : clusters) {
        if (c.points.empty()) continue;
        Point ctr;
        for (const auto& p : c.points) {
            ctr.x += p.x;
            ctr.y += p.y;
            ctr.z += p.z;
            ctr.doppler += p.doppler;
            ctr.snr += p.snr;
        }
        const float inv = 1.0f / static_cast<float>(c.points.size());
        ctr.x *= inv;
        ctr.y *= inv;
        ctr.z *= inv;
        ctr.doppler *= inv;
        ctr.snr *= inv;
        c.centroid = ctr;
    }

    return clusters;
}

}  // namespace mmwave

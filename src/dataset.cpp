#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <unordered_map>

namespace mmwave {

namespace {
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(line);
    while (std::getline(iss, cur, ',')) out.push_back(cur);
    return out;
}
}  // namespace

bool DatasetLoader::load_from_manifest(const std::string& manifest_path, Dataset& out_dataset) const {
    out_dataset.samples.clear();
    std::ifstream ifs(manifest_path);
    if (!ifs.is_open()) return false;

    std::string line;
    bool is_header = true;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (is_header) {
            is_header = false;
            if (line.find("sequence_id") != std::string::npos) continue;
        }
        const auto cols = split_csv(line);
        if (cols.size() < 3) continue;

        Sequence seq;
        seq.id = cols[0];
        seq.label = std::stoi(cols[1]);
        const std::string seq_file = cols[2];

        std::ifstream sfs(seq_file);
        if (!sfs.is_open()) continue;

        std::unordered_map<int, Frame> frame_map;
        std::string seq_line;
        bool seq_header = true;
        while (std::getline(sfs, seq_line)) {
            if (seq_line.empty()) continue;
            if (seq_header) {
                seq_header = false;
                if (seq_line.find("frame_idx") != std::string::npos) continue;
            }
            const auto p = split_csv(seq_line);
            if (p.size() < 7) continue;
            const int frame_idx = std::stoi(p[0]);
            Frame& f = frame_map[frame_idx];
            f.timestamp = std::stoll(p[1]);
            Point pt;
            pt.x = std::stof(p[2]);
            pt.y = std::stof(p[3]);
            pt.z = std::stof(p[4]);
            pt.doppler = std::stof(p[5]);
            pt.snr = std::stof(p[6]);
            f.points.push_back(pt);
        }

        std::vector<std::pair<int, Frame>> ordered(frame_map.begin(), frame_map.end());
        std::sort(ordered.begin(), ordered.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        seq.frames.reserve(ordered.size());
        for (auto& kv : ordered) seq.frames.push_back(std::move(kv.second));

        if (!seq.frames.empty()) out_dataset.samples.push_back(std::move(seq));
    }

    return !out_dataset.samples.empty();
}

SplitData split_dataset(const Dataset& dataset, float test_ratio, unsigned int seed) {
    SplitData out;
    if (dataset.samples.empty()) return out;
    std::vector<std::size_t> idx(dataset.samples.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    const std::size_t test_size =
        static_cast<std::size_t>(static_cast<float>(idx.size()) * test_ratio);
    for (std::size_t i = 0; i < idx.size(); ++i) {
        if (i < test_size) {
            out.test.push_back(dataset.samples[idx[i]]);
        } else {
            out.train.push_back(dataset.samples[idx[i]]);
        }
    }
    return out;
}

}  // namespace mmwave

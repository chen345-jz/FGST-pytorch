#include "dataset.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>

namespace mmwave {

namespace {
bool parse_npy_header(std::ifstream& ifs, std::vector<int>& shape, std::string& descr) {
    char magic[6] = {};
    ifs.read(magic, 6);
    if (ifs.gcount() != 6) return false;
    if (!(magic[0] == char(0x93) && magic[1] == 'N' && magic[2] == 'U' && magic[3] == 'M' &&
          magic[4] == 'P' && magic[5] == 'Y')) {
        return false;
    }

    char major = 0, minor = 0;
    ifs.read(&major, 1);
    ifs.read(&minor, 1);
    if (!ifs.good()) return false;

    std::uint32_t header_len = 0;
    if (major == 1) {
        std::uint16_t h16 = 0;
        ifs.read(reinterpret_cast<char*>(&h16), 2);
        header_len = h16;
    } else {
        ifs.read(reinterpret_cast<char*>(&header_len), 4);
    }
    if (!ifs.good() || header_len == 0) return false;

    std::string header(header_len, '\0');
    ifs.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!ifs.good()) return false;

    std::smatch m_descr;
    std::regex r_descr("'descr'\\s*:\\s*'([^']+)'");
    if (!std::regex_search(header, m_descr, r_descr) || m_descr.size() < 2) return false;
    descr = m_descr[1];

    std::smatch m_shape;
    std::regex r_shape("'shape'\\s*:\\s*\\(([^\\)]*)\\)");
    if (!std::regex_search(header, m_shape, r_shape) || m_shape.size() < 2) return false;
    std::string shape_txt = m_shape[1];
    std::replace(shape_txt.begin(), shape_txt.end(), ',', ' ');
    std::istringstream iss(shape_txt);
    int v = 0;
    shape.clear();
    while (iss >> v) shape.push_back(v);

    return !shape.empty();
}
}  // namespace

bool DatasetLoader::load_from_npy_root(const std::string& root_path, Dataset& out_dataset) const {
    namespace fs = std::filesystem;
    out_dataset.samples.clear();
    out_dataset.label_names.clear();

    const fs::path root(root_path);
    if (!fs::exists(root) || !fs::is_directory(root)) return false;

    std::vector<fs::path> person_dirs;
    for (const auto& e : fs::directory_iterator(root)) {
        if (e.is_directory()) person_dirs.push_back(e.path());
    }
    std::sort(person_dirs.begin(), person_dirs.end());

    for (const auto& pd : person_dirs) {
        const std::string name = pd.filename().string();
        if (name.rfind("p_", 0) != 0) continue;
        int label = -1;
        try {
            label = std::stoi(name.substr(2));
        } catch (...) {
            continue;
        }
        out_dataset.label_names[label] = name;

        std::vector<fs::path> npy_files;
        for (const auto& f : fs::directory_iterator(pd)) {
            if (f.is_regular_file() && f.path().extension() == ".npy") npy_files.push_back(f.path());
        }
        std::sort(npy_files.begin(), npy_files.end());

        for (const auto& f : npy_files) {
            std::ifstream ifs(f, std::ios::binary);
            if (!ifs.is_open()) continue;

            std::vector<int> shape;
            std::string descr;
            if (!parse_npy_header(ifs, shape, descr)) continue;
            if (shape.size() != 3 || shape[2] < 4) continue;
            if (!(descr == "<f8" || descr == "|f8" || descr == "<f4" || descr == "|f4")) continue;

            const int T = shape[0];
            const int P = shape[1];
            const int C = shape[2];
            Sequence seq;
            seq.id = name + "_" + f.stem().string();
            seq.label = label;
            seq.frames.resize(T);

            const bool is_f64 = (descr.find("f8") != std::string::npos);
            for (int t = 0; t < T; ++t) {
                Frame frame;
                frame.timestamp = static_cast<std::int64_t>(t) * 50;
                frame.points.reserve(P);
                for (int p = 0; p < P; ++p) {
                    float vals[4] = {0, 0, 0, 0};
                    for (int c = 0; c < C; ++c) {
                        if (is_f64) {
                            double dv = 0.0;
                            ifs.read(reinterpret_cast<char*>(&dv), sizeof(double));
                            if (c < 4) vals[c] = static_cast<float>(dv);
                        } else {
                            float fv = 0.0f;
                            ifs.read(reinterpret_cast<char*>(&fv), sizeof(float));
                            if (c < 4) vals[c] = fv;
                        }
                    }
                    Point pt;
                    pt.x = vals[0];
                    pt.y = vals[1];
                    pt.z = vals[2];
                    pt.doppler = vals[3];
                    pt.snr = 10.0f;
                    frame.points.push_back(pt);
                }
                if (!ifs.good()) break;
                seq.frames[t] = std::move(frame);
            }

            if (!seq.frames.empty()) out_dataset.samples.push_back(std::move(seq));
        }
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

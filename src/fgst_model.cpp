#include "fgst_model.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <utility>

#ifdef MMWAVE_USE_LIBTORCH
#include <torch/torch.h>
#endif

namespace mmwave {

bool FGSTModel::available() const {
#ifdef MMWAVE_USE_LIBTORCH
    return true;
#else
    return false;
#endif
}

#ifdef MMWAVE_USE_LIBTORCH
namespace {

struct FgstNetImpl : torch::nn::Module {
    FgstNetImpl(int in_dim, int point_dim, int temporal_dim, int num_parts, int num_classes)
        : num_parts_(num_parts),
          point_mlp(register_module("point_mlp", torch::nn::Sequential(
                                                 torch::nn::Linear(in_dim, point_dim),
                                                 torch::nn::ReLU(),
                                                 torch::nn::Linear(point_dim, point_dim),
                                                 torch::nn::ReLU()))),
          part_prob(register_module("part_prob", torch::nn::Linear(point_dim, num_parts))),
          part_temporal(register_module("part_temporal", torch::nn::Sequential(
                                                    torch::nn::Conv1d(torch::nn::Conv1dOptions(point_dim, temporal_dim, 3).padding(1)),
                                                    torch::nn::ReLU(),
                                                    torch::nn::Conv1d(torch::nn::Conv1dOptions(temporal_dim, temporal_dim, 3).padding(1)),
                                                    torch::nn::ReLU()))),
          global_temporal(register_module("global_temporal", torch::nn::Sequential(
                                                      torch::nn::Conv1d(torch::nn::Conv1dOptions(point_dim, temporal_dim, 3).padding(1)),
                                                      torch::nn::ReLU(),
                                                      torch::nn::Conv1d(torch::nn::Conv1dOptions(temporal_dim, temporal_dim, 3).padding(1)),
                                                      torch::nn::ReLU()))),
          fusion(register_module("fusion", torch::nn::Sequential(
                                             torch::nn::Linear((num_parts + 1) * temporal_dim, temporal_dim),
                                             torch::nn::ReLU(),
                                             torch::nn::Dropout(0.2),
                                             torch::nn::Linear(temporal_dim, num_classes)))) {}

    torch::Tensor forward(torch::Tensor x) {
        // x: [B, T, P, C]
        const auto B = x.size(0);
        const auto T = x.size(1);
        const auto P = x.size(2);
        const auto C = x.size(3);

        auto xp = x.view({B * T * P, C});
        auto point_feat = point_mlp->forward(xp);                      // [B*T*P, F]
        auto prob = torch::softmax(part_prob->forward(point_feat), 1); // [B*T*P, K]
        point_feat = point_feat.view({B, T, P, -1});                  // [B, T, P, F]
        prob = prob.view({B, T, P, num_parts_});                      // [B, T, P, K]

        // Global branch: max pooling across points per frame.
        auto global_frame = std::get<0>(point_feat.max(2));           // [B, T, F]
        auto global_ts = global_frame.transpose(1, 2);                // [B, F, T]
        global_ts = global_temporal->forward(global_ts);              // [B, H, T]
        auto global_vec = std::get<0>(global_ts.max(2));              // [B, H]

        // Part-guided local branch.
        std::vector<torch::Tensor> part_vecs;
        part_vecs.reserve(num_parts_);
        for (int k = 0; k < num_parts_; ++k) {
            auto wk = prob.select(3, k).unsqueeze(-1);                // [B, T, P, 1]
            auto weighted = point_feat * wk;                          // [B, T, P, F]
            auto pooled = weighted.sum(2) / (wk.sum(2) + 1e-6);       // [B, T, F]
            auto ts = pooled.transpose(1, 2);                         // [B, F, T]
            ts = part_temporal->forward(ts);                          // [B, H, T]
            part_vecs.push_back(std::get<0>(ts.max(2)));              // [B, H]
        }

        auto part_cat = torch::cat(part_vecs, 1);                     // [B, K*H]
        auto fused = torch::cat({part_cat, global_vec}, 1);           // [B, (K+1)*H]
        return fusion->forward(fused);                                // [B, num_classes]
    }

    int num_parts_ = 4;
    torch::nn::Sequential point_mlp{nullptr};
    torch::nn::Linear part_prob{nullptr};
    torch::nn::Sequential part_temporal{nullptr};
    torch::nn::Sequential global_temporal{nullptr};
    torch::nn::Sequential fusion{nullptr};
};
TORCH_MODULE(FgstNet);

torch::Tensor seq_to_tensor(const Sequence& seq, int max_t, int max_p) {
    auto x = torch::zeros({max_t, max_p, 5}, torch::kFloat32);
    const int t_lim = std::min(max_t, static_cast<int>(seq.frames.size()));
    for (int t = 0; t < t_lim; ++t) {
        const auto& pts = seq.frames[t].points;
        const int p_lim = std::min(max_p, static_cast<int>(pts.size()));
        for (int p = 0; p < p_lim; ++p) {
            x[t][p][0] = pts[p].x;
            x[t][p][1] = pts[p].y;
            x[t][p][2] = pts[p].z;
            x[t][p][3] = pts[p].doppler;
            x[t][p][4] = pts[p].snr;
        }
    }
    return x;
}

}  // namespace
#endif

bool FGSTModel::fit(const std::vector<Sequence>& train_data, const std::vector<int>& train_labels) {
#ifdef MMWAVE_USE_LIBTORCH
    if (train_data.empty() || train_data.size() != train_labels.size()) return false;
    labels_ = train_labels;
    std::sort(labels_.begin(), labels_.end());
    labels_.erase(std::unique(labels_.begin(), labels_.end()), labels_.end());
    if (labels_.size() < 2) return false;

    std::unordered_map<int, int> lab2idx;
    for (int i = 0; i < static_cast<int>(labels_.size()); ++i) lab2idx[labels_[i]] = i;

    FgstNet net(/*in_dim=*/5, cfg_.point_feature_dim, cfg_.temporal_feature_dim, cfg_.num_body_parts,
                static_cast<int>(labels_.size()));
    net->train();
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(cfg_.learning_rate).weight_decay(cfg_.weight_decay));

    const int n = static_cast<int>(train_data.size());
    for (int ep = 0; ep < cfg_.epochs; ++ep) {
        float ep_loss = 0.0f;
        for (int i = 0; i < n; i += cfg_.batch_size) {
            const int b = std::min(cfg_.batch_size, n - i);
            std::vector<torch::Tensor> xs;
            std::vector<int64_t> ys;
            xs.reserve(b);
            ys.reserve(b);
            for (int j = 0; j < b; ++j) {
                xs.push_back(seq_to_tensor(train_data[i + j], cfg_.max_frames, cfg_.max_points_per_frame));
                ys.push_back(static_cast<int64_t>(lab2idx[train_labels[i + j]]));
            }
            auto x = torch::stack(xs, 0);
            auto y = torch::tensor(ys, torch::kLong);
            auto logits = net->forward(x);
            auto loss = torch::nn::functional::cross_entropy(logits, y);
            opt.zero_grad();
            loss.backward();
            opt.step();
            ep_loss += loss.item<float>();
        }
        if ((ep + 1) % 10 == 0 || ep == 0) {
            std::cout << "[fgst] epoch " << (ep + 1) << "/" << cfg_.epochs << " loss=" << ep_loss << "\n";
        }
    }

    last_saved_path_ = ".tmp_fgst_model.pt";
    torch::save(net, last_saved_path_);
    return true;
#else
    std::cerr << "PointNet++/FGST requires LibTorch. Reconfigure with -DUSE_LIBTORCH=ON.\n";
    (void)train_data;
    (void)train_labels;
    return false;
#endif
}

int FGSTModel::predict_one(const Sequence&) const {
#ifdef MMWAVE_USE_LIBTORCH
    return -1;
#else
    return -1;
#endif
}

std::vector<int> FGSTModel::predict(const std::vector<Sequence>& data) const {
#ifdef MMWAVE_USE_LIBTORCH
    if (last_saved_path_.empty() || labels_.empty()) return {};
    FgstNet net(/*in_dim=*/5, cfg_.point_feature_dim, cfg_.temporal_feature_dim, cfg_.num_body_parts,
                static_cast<int>(labels_.size()));
    torch::load(net, last_saved_path_);
    net->eval();

    std::vector<int> out;
    out.reserve(data.size());
    for (const auto& s : data) {
        auto x = seq_to_tensor(s, cfg_.max_frames, cfg_.max_points_per_frame).unsqueeze(0);
        auto logits = net->forward(x);
        const int idx = logits.argmax(1).item<int>();
        out.push_back(labels_[idx]);
    }
    return out;
#else
    std::vector<int> out;
    out.reserve(data.size());
    for (const auto& s : data) out.push_back(predict_one(s));
    return out;
#endif
}

bool FGSTModel::save(const std::string& path) const {
#ifdef MMWAVE_USE_LIBTORCH
    if (last_saved_path_.empty()) return false;
    std::ifstream ifs(last_saved_path_, std::ios::binary);
    std::ofstream ofs(path, std::ios::binary);
    if (!ifs.is_open() || !ofs.is_open()) return false;
    ofs << ifs.rdbuf();
    std::ofstream m(path + ".labels");
    if (!m.is_open()) return false;
    m << cfg_.epochs << "," << cfg_.batch_size << "," << cfg_.learning_rate << "," << cfg_.weight_decay << ","
      << cfg_.max_frames << "," << cfg_.max_points_per_frame << "," << cfg_.num_body_parts << ","
      << cfg_.point_feature_dim << "," << cfg_.temporal_feature_dim << "\n";
    for (std::size_t i = 0; i < labels_.size(); ++i) {
        if (i) m << ",";
        m << labels_[i];
    }
    return ofs.good() && m.good();
#else
    (void)path;
    return false;
#endif
}

bool FGSTModel::load(const std::string& path) {
#ifdef MMWAVE_USE_LIBTORCH
    std::ifstream m(path + ".labels");
    if (!m.is_open()) return false;
    std::string line1, line2;
    std::getline(m, line1);
    std::getline(m, line2);
    if (line1.empty() || line2.empty()) return false;
    {
        std::replace(line1.begin(), line1.end(), ',', ' ');
        std::istringstream iss(line1);
        iss >> cfg_.epochs >> cfg_.batch_size >> cfg_.learning_rate >> cfg_.weight_decay >> cfg_.max_frames >>
            cfg_.max_points_per_frame >> cfg_.num_body_parts >> cfg_.point_feature_dim >> cfg_.temporal_feature_dim;
    }
    labels_.clear();
    {
        std::replace(line2.begin(), line2.end(), ',', ' ');
        std::istringstream iss(line2);
        int v = -1;
        while (iss >> v) labels_.push_back(v);
    }
    last_saved_path_ = path;
    return true;
#else
    (void)path;
    return false;
#endif
}

}  // namespace mmwave

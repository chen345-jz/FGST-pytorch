#include "model_io.hpp"

#include <fstream>

namespace mmwave {

bool save_model(const std::string& path, const KNNClassifier& model) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;

    const auto& x = model.train_x();
    const auto& y = model.train_y();
    const int k = model.config().k_neighbors;
    const std::size_t rows = x.size();
    const std::size_t cols = rows ? x[0].size() : 0;
    ofs.write(reinterpret_cast<const char*>(&k), sizeof(k));
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    for (std::size_t i = 0; i < rows; ++i) {
        ofs.write(reinterpret_cast<const char*>(x[i].data()), sizeof(float) * cols);
        ofs.write(reinterpret_cast<const char*>(&y[i]), sizeof(int));
    }
    return ofs.good();
}

bool load_model(const std::string& path, KNNClassifier& model) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;

    int k = 3;
    std::size_t rows = 0, cols = 0;
    ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!ifs.good()) return false;

    FeatureMatrix x(rows, FeatureVector(cols, 0.0f));
    std::vector<int> y(rows, -1);
    for (std::size_t i = 0; i < rows; ++i) {
        ifs.read(reinterpret_cast<char*>(x[i].data()), sizeof(float) * cols);
        ifs.read(reinterpret_cast<char*>(&y[i]), sizeof(int));
        if (!ifs.good()) return false;
    }

    ClassifierConfig cfg;
    cfg.k_neighbors = k;
    KNNClassifier loaded(cfg);
    loaded.fit(x, y);
    model = loaded;
    return true;
}

bool save_model(const std::string& path, const SoftmaxClassifier& model) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) return false;
    const int cls = model.num_classes();
    const int feat = model.num_features();
    const int epochs = model.config().epochs;
    const float lr = model.config().learning_rate;
    const float l2 = model.config().l2;
    const std::size_t nlab = model.labels().size();
    const std::size_t nw = model.weights().size();
    ofs.write(reinterpret_cast<const char*>(&cls), sizeof(cls));
    ofs.write(reinterpret_cast<const char*>(&feat), sizeof(feat));
    ofs.write(reinterpret_cast<const char*>(&epochs), sizeof(epochs));
    ofs.write(reinterpret_cast<const char*>(&lr), sizeof(lr));
    ofs.write(reinterpret_cast<const char*>(&l2), sizeof(l2));
    ofs.write(reinterpret_cast<const char*>(&nlab), sizeof(nlab));
    ofs.write(reinterpret_cast<const char*>(model.labels().data()), sizeof(int) * nlab);
    ofs.write(reinterpret_cast<const char*>(&nw), sizeof(nw));
    ofs.write(reinterpret_cast<const char*>(model.weights().data()), sizeof(float) * nw);
    return ofs.good();
}

bool load_model(const std::string& path, SoftmaxClassifier& model) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    int cls = 0;
    int feat = 0;
    int epochs = 200;
    float lr = 0.05f;
    float l2 = 1e-4f;
    std::size_t nlab = 0;
    std::size_t nw = 0;
    ifs.read(reinterpret_cast<char*>(&cls), sizeof(cls));
    ifs.read(reinterpret_cast<char*>(&feat), sizeof(feat));
    ifs.read(reinterpret_cast<char*>(&epochs), sizeof(epochs));
    ifs.read(reinterpret_cast<char*>(&lr), sizeof(lr));
    ifs.read(reinterpret_cast<char*>(&l2), sizeof(l2));
    ifs.read(reinterpret_cast<char*>(&nlab), sizeof(nlab));
    if (!ifs.good()) return false;
    std::vector<int> labels(nlab, -1);
    ifs.read(reinterpret_cast<char*>(labels.data()), sizeof(int) * nlab);
    ifs.read(reinterpret_cast<char*>(&nw), sizeof(nw));
    std::vector<float> w(nw, 0.0f);
    ifs.read(reinterpret_cast<char*>(w.data()), sizeof(float) * nw);
    if (!ifs.good()) return false;
    SoftmaxConfig cfg;
    cfg.epochs = epochs;
    cfg.learning_rate = lr;
    cfg.l2 = l2;
    model.load_state(std::move(labels), std::move(w), cls, feat, cfg);
    return true;
}

}  // namespace mmwave

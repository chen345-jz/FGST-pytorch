#include "pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

#include "classifier.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "features.hpp"
#include "fgst_model.hpp"
#include "metrics.hpp"
#include "model_io.hpp"
#include "preprocess.hpp"
#include "softmax_classifier.hpp"
#include "types.hpp"

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

    std::uint32_t hlen = 0;
    if (major == 1) {
        std::uint16_t h16 = 0;
        ifs.read(reinterpret_cast<char*>(&h16), 2);
        hlen = h16;
    } else {
        ifs.read(reinterpret_cast<char*>(&hlen), 4);
    }
    if (!ifs.good() || hlen == 0) return false;

    std::string header(hlen, '\0');
    ifs.read(header.data(), static_cast<std::streamsize>(hlen));
    if (!ifs.good()) return false;

    std::smatch m1, m2;
    std::regex r1("'descr'\\s*:\\s*'([^']+)'");
    std::regex r2("'shape'\\s*:\\s*\\(([^\\)]*)\\)");
    if (!std::regex_search(header, m1, r1) || !std::regex_search(header, m2, r2)) return false;
    descr = m1[1];
    std::string s = m2[1];
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream iss(s);
    shape.clear();
    int v = 0;
    while (iss >> v) shape.push_back(v);
    return !shape.empty();
}

bool load_one_sequence_npy(const std::string& path, Sequence& seq) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) return false;
    std::vector<int> shape;
    std::string descr;
    if (!parse_npy_header(ifs, shape, descr)) return false;
    if (shape.size() != 3 || shape[2] < 4) return false;
    const int T = shape[0];
    const int P = shape[1];
    const int C = shape[2];
    const bool is_f64 = (descr.find("f8") != std::string::npos);

    seq.id = "predict_sample";
    seq.label = -1;
    seq.frames.clear();
    seq.frames.resize(T);
    for (int t = 0; t < T; ++t) {
        Frame frame;
        frame.timestamp = static_cast<std::int64_t>(t) * 50;
        frame.points.reserve(P);
        for (int p = 0; p < P; ++p) {
            float vals[4] = {0, 0, 0, 0};
            for (int c = 0; c < C; ++c) {
                if (is_f64) {
                    double d = 0.0;
                    ifs.read(reinterpret_cast<char*>(&d), sizeof(double));
                    if (c < 4) vals[c] = static_cast<float>(d);
                } else {
                    float f = 0.0f;
                    ifs.read(reinterpret_cast<char*>(&f), sizeof(float));
                    if (c < 4) vals[c] = f;
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
        if (!ifs.good()) return false;
        seq.frames[t] = std::move(frame);
    }
    return true;
}

PreprocessParams read_preprocess(const Config& cfg) {
    PreprocessParams p;
    p.min_snr = cfg.get_float("preprocess.min_snr", 0.0f);
    p.min_range = cfg.get_float("preprocess.min_range", 0.0f);
    p.max_range = cfg.get_float("preprocess.max_range", 20.0f);
    p.min_points_per_frame = static_cast<std::size_t>(cfg.get_int("preprocess.min_points_per_frame", 1));
    return p;
}

FGSTConfig read_fgst_config(const Config& cfg) {
    FGSTConfig c;
    c.epochs = cfg.get_int("fgst.epochs", 50);
    c.batch_size = cfg.get_int("fgst.batch_size", 16);
    c.learning_rate = cfg.get_float("fgst.learning_rate", 1e-3f);
    c.weight_decay = cfg.get_float("fgst.weight_decay", 1e-4f);
    c.max_frames = cfg.get_int("fgst.max_frames", 20);
    c.max_points_per_frame = cfg.get_int("fgst.max_points_per_frame", 80);
    c.num_body_parts = cfg.get_int("fgst.num_body_parts", 4);
    c.point_feature_dim = cfg.get_int("fgst.point_feature_dim", 64);
    c.temporal_feature_dim = cfg.get_int("fgst.temporal_feature_dim", 128);
    return c;
}
}  // namespace

int run_train(const std::string& config_path) {
    Config cfg;
    if (!cfg.load(config_path)) return 1;

    DatasetLoader loader;
    Dataset dataset;
    const std::string npy_root = cfg.get("data.npy_root_path", "./2s/2s");
    if (!loader.load_from_npy_root(npy_root, dataset)) {
        std::cerr << "failed to load npy dataset from root: " << npy_root << "\n";
        return 2;
    }

    const float test_ratio = cfg.get_float("data.test_ratio", 0.2f);
    const unsigned seed = static_cast<unsigned>(cfg.get_int("data.seed", 42));
    auto split = split_dataset(dataset, test_ratio, seed);

    Preprocessor preprocessor(read_preprocess(cfg));
    FeatureExtractor extractor;
    const float eps = cfg.get_float("cluster.eps", 0.25f);
    const std::size_t min_pts = static_cast<std::size_t>(cfg.get_int("cluster.min_pts", 5));

    FeatureMatrix x_train, x_test;
    std::vector<int> y_train, y_test;
    std::vector<Sequence> seq_train, seq_test;
    for (auto seq : split.train) {
        preprocessor.filter_sequence(seq);
        seq_train.push_back(seq);
        x_train.push_back(extractor.extract(seq, preprocessor, eps, min_pts));
        y_train.push_back(seq.label);
    }
    for (auto seq : split.test) {
        preprocessor.filter_sequence(seq);
        seq_test.push_back(seq);
        x_test.push_back(extractor.extract(seq, preprocessor, eps, min_pts));
        y_test.push_back(seq.label);
    }

    const std::string model_type = cfg.get("classifier.type", "fgst");
    const std::string model_path = cfg.get("model.path", "./model/mmwave_fgst_2s.pt");
    const std::string report_path = cfg.get("report.path", "./model/metrics_fgst_2s.csv");
    std::vector<int> train_pred, test_pred;

    if (model_type == "fgst") {
        FGSTModel model(read_fgst_config(cfg));
        if (!model.fit(seq_train, y_train)) return 4;
        train_pred = model.predict(seq_train);
        test_pred = model.predict(seq_test);
        if (!model.save(model_path)) return 3;
    } else if (model_type == "softmax") {
        SoftmaxConfig s_cfg;
        s_cfg.learning_rate = cfg.get_float("softmax.learning_rate", 0.05f);
        s_cfg.epochs = cfg.get_int("softmax.epochs", 200);
        s_cfg.l2 = cfg.get_float("softmax.l2", 1e-4f);
        SoftmaxClassifier model(s_cfg);
        model.fit(x_train, y_train);
        train_pred = model.predict(x_train);
        test_pred = model.predict(x_test);
        if (!save_model(model_path, model)) return 3;
    } else {
        ClassifierConfig cls_cfg;
        cls_cfg.k_neighbors = cfg.get_int("classifier.k_neighbors", 3);
        KNNClassifier model(cls_cfg);
        model.fit(x_train, y_train);
        train_pred = model.predict(x_train);
        test_pred = model.predict(x_test);
        if (!save_model(model_path, model)) return 3;
    }

    std::cout << "train samples: " << x_train.size() << ", test samples: " << x_test.size() << "\n";
    if (!train_pred.empty()) {
        std::cout << "[train]\n";
        print_metrics(compute_metrics(y_train, train_pred));
    }
    if (!test_pred.empty()) {
        auto te = compute_metrics(y_test, test_pred);
        std::cout << "[test]\n";
        print_metrics(te);
        save_metrics_csv(report_path, te);
        std::cout << "metrics saved: " << report_path << "\n";
    }
    std::cout << "model saved: " << model_path << "\n";
    return 0;
}

int run_eval(const std::string& config_path) {
    Config cfg;
    if (!cfg.load(config_path)) return 1;

    DatasetLoader loader;
    Dataset dataset;
    const std::string npy_root = cfg.get("data.npy_root_path", "./2s/2s");
    if (!loader.load_from_npy_root(npy_root, dataset)) {
        std::cerr << "failed to load npy dataset from root: " << npy_root << "\n";
        return 3;
    }

    const std::string model_path = cfg.get("model.path", "./model/mmwave_fgst_2s.pt");
    const std::string model_type = cfg.get("classifier.type", "fgst");

    Preprocessor preprocessor(read_preprocess(cfg));
    FeatureExtractor extractor;
    const float eps = cfg.get_float("cluster.eps", 0.25f);
    const std::size_t min_pts = static_cast<std::size_t>(cfg.get_int("cluster.min_pts", 5));

    FeatureMatrix x;
    std::vector<int> y;
    std::vector<Sequence> seqs;
    for (auto seq : dataset.samples) {
        preprocessor.filter_sequence(seq);
        seqs.push_back(seq);
        x.push_back(extractor.extract(seq, preprocessor, eps, min_pts));
        y.push_back(seq.label);
    }

    std::vector<int> pred;
    if (model_type == "fgst") {
        FGSTModel model(read_fgst_config(cfg));
        if (!model.load(model_path)) return 2;
        pred = model.predict(seqs);
    } else if (model_type == "softmax") {
        SoftmaxClassifier model(SoftmaxConfig{});
        if (!load_model(model_path, model)) return 2;
        pred = model.predict(x);
    } else {
        ClassifierConfig cls_cfg;
        cls_cfg.k_neighbors = 3;
        KNNClassifier model(cls_cfg);
        if (!load_model(model_path, model)) return 2;
        pred = model.predict(x);
    }

    std::cout << "eval samples: " << x.size() << "\n";
    auto report = compute_metrics(y, pred);
    print_metrics(report);
    const std::string report_path = cfg.get("report.path", "./model/metrics_fgst_2s.csv");
    save_metrics_csv(report_path, report);
    std::cout << "metrics saved: " << report_path << "\n";
    return 0;
}

int run_predict(const std::string& config_path, const std::string& sample_npy_path) {
    Config cfg;
    if (!cfg.load(config_path)) return 1;

    Sequence seq;
    if (!load_one_sequence_npy(sample_npy_path, seq)) {
        std::cerr << "failed to load sample npy: " << sample_npy_path << "\n";
        return 3;
    }

    const std::string model_path = cfg.get("model.path", "./model/mmwave_fgst_2s.pt");
    const std::string model_type = cfg.get("classifier.type", "fgst");

    Preprocessor preprocessor(read_preprocess(cfg));
    preprocessor.filter_sequence(seq);
    FeatureExtractor extractor;
    const float eps = cfg.get_float("cluster.eps", 0.25f);
    const std::size_t min_pts = static_cast<std::size_t>(cfg.get_int("cluster.min_pts", 5));
    const auto feat = extractor.extract(seq, preprocessor, eps, min_pts);

    int pred = -1;
    if (model_type == "fgst") {
        FGSTModel model(read_fgst_config(cfg));
        if (!model.load(model_path)) return 2;
        auto p = model.predict(std::vector<Sequence>{seq});
        pred = p.empty() ? -1 : p[0];
    } else if (model_type == "softmax") {
        SoftmaxClassifier model(SoftmaxConfig{});
        if (!load_model(model_path, model)) return 2;
        pred = model.predict_one(feat);
    } else {
        ClassifierConfig cls_cfg;
        cls_cfg.k_neighbors = 3;
        KNNClassifier model(cls_cfg);
        if (!load_model(model_path, model)) return 2;
        pred = model.predict_one(feat);
    }

    std::cout << "predict label: " << pred << "\n";
    const std::string out = cfg.get("predict.output_path", "./model/predict_result_2s.csv");
    std::ofstream ofs(out);
    if (ofs.is_open()) {
        ofs << "sample_npy,predict_label\n";
        ofs << sample_npy_path << "," << pred << "\n";
        std::cout << "predict saved: " << out << "\n";
    }
    return 0;
}

}  // namespace mmwave

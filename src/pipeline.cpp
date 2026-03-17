#include "pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "classifier.hpp"
#include "config.hpp"
#include "dataset.hpp"
#include "features.hpp"
#include "metrics.hpp"
#include "model_io.hpp"
#include "preprocess.hpp"
#include "fgst_model.hpp"
#include "softmax_classifier.hpp"
#include "types.hpp"

namespace mmwave {

namespace {
bool load_one_sequence_csv(const std::string& path, Sequence& seq) {
    std::ifstream sfs(path);
    if (!sfs.is_open()) return false;
    seq.id = "predict_sample";
    seq.label = -1;
    std::unordered_map<int, Frame> frame_map;
    std::string line;
    bool is_header = true;
    while (std::getline(sfs, line)) {
        if (line.empty()) continue;
        if (is_header) {
            is_header = false;
            if (line.find("frame_idx") != std::string::npos) continue;
        }
        std::vector<std::string> cols;
        std::string cur;
        std::istringstream iss(line);
        while (std::getline(iss, cur, ',')) cols.push_back(cur);
        if (cols.size() < 7) continue;
        int idx = std::stoi(cols[0]);
        Frame& f = frame_map[idx];
        f.timestamp = std::stoll(cols[1]);
        Point p;
        p.x = std::stof(cols[2]);
        p.y = std::stof(cols[3]);
        p.z = std::stof(cols[4]);
        p.doppler = std::stof(cols[5]);
        p.snr = std::stof(cols[6]);
        f.points.push_back(p);
    }
    std::vector<std::pair<int, Frame>> ordered(frame_map.begin(), frame_map.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });
    for (auto& kv : ordered) seq.frames.push_back(std::move(kv.second));
    return !seq.frames.empty();
}

PreprocessParams read_preprocess(const Config& cfg) {
    PreprocessParams p;
    p.min_snr = cfg.get_float("preprocess.min_snr", 0.0f);
    p.min_range = cfg.get_float("preprocess.min_range", 0.0f);
    p.max_range = cfg.get_float("preprocess.max_range", 8.0f);
    p.min_points_per_frame = static_cast<std::size_t>(cfg.get_int("preprocess.min_points_per_frame", 5));
    return p;
}
}  // namespace

int run_train(const std::string& config_path) {
    Config cfg;
    if (!cfg.load(config_path)) {
        std::cerr << "failed to load config: " << config_path << "\n";
        return 1;
    }

    DatasetLoader loader;
    Dataset dataset;
    const std::string manifest = cfg.get("data.manifest_path");
    if (!loader.load_from_manifest(manifest, dataset)) {
        std::cerr << "failed to load dataset from manifest: " << manifest << "\n";
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

    const std::string model_type = cfg.get("classifier.type", "knn");
    const std::string model_path = cfg.get("model.path", "model.bin");
    const std::string report_path = cfg.get("report.path", "./model/metrics_train.csv");
    std::vector<int> train_pred;
    std::vector<int> test_pred;

    if (model_type == "fgst") {
        FGSTConfig p_cfg;
        p_cfg.epochs = cfg.get_int("fgst.epochs", 50);
        p_cfg.batch_size = cfg.get_int("fgst.batch_size", 16);
        p_cfg.learning_rate = cfg.get_float("fgst.learning_rate", 1e-3f);
        p_cfg.weight_decay = cfg.get_float("fgst.weight_decay", 1e-4f);
        p_cfg.max_frames = cfg.get_int("fgst.max_frames", 32);
        p_cfg.max_points_per_frame =
            cfg.get_int("fgst.max_points_per_frame", 64);
        p_cfg.num_body_parts = cfg.get_int("fgst.num_body_parts", 4);
        p_cfg.point_feature_dim =
            cfg.get_int("fgst.point_feature_dim", 64);
        p_cfg.temporal_feature_dim =
            cfg.get_int("fgst.temporal_feature_dim", 128);
        FGSTModel model(p_cfg);
        if (!model.fit(seq_train, y_train)) {
            std::cerr << "failed to train fgst model.\n";
            return 4;
        }
        train_pred = model.predict(seq_train);
        test_pred = model.predict(seq_test);
        if (!model.save(model_path)) {
            std::cerr << "failed to save fgst model: " << model_path << "\n";
            return 3;
        }
    } else if (model_type == "softmax") {
        SoftmaxConfig s_cfg;
        s_cfg.learning_rate = cfg.get_float("softmax.learning_rate", 0.05f);
        s_cfg.epochs = cfg.get_int("softmax.epochs", 200);
        s_cfg.l2 = cfg.get_float("softmax.l2", 1e-4f);
        SoftmaxClassifier model(s_cfg);
        model.fit(x_train, y_train);
        train_pred = model.predict(x_train);
        test_pred = model.predict(x_test);
        if (!save_model(model_path, model)) {
            std::cerr << "failed to save model: " << model_path << "\n";
            return 3;
        }
    } else {
        const int k = cfg.get_int("classifier.k_neighbors", 3);
        ClassifierConfig cls_cfg;
        cls_cfg.k_neighbors = k;
        KNNClassifier model(cls_cfg);
        model.fit(x_train, y_train);
        train_pred = model.predict(x_train);
        test_pred = model.predict(x_test);
        if (!save_model(model_path, model)) {
            std::cerr << "failed to save model: " << model_path << "\n";
            return 3;
        }
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
    if (!cfg.load(config_path)) {
        std::cerr << "failed to load config: " << config_path << "\n";
        return 1;
    }

    const std::string model_path = cfg.get("model.path", "model.bin");
    const std::string model_type = cfg.get("classifier.type", "knn");

    DatasetLoader loader;
    Dataset dataset;
    const std::string manifest = cfg.get("data.manifest_path");
    if (!loader.load_from_manifest(manifest, dataset)) {
        std::cerr << "failed to load dataset from manifest: " << manifest << "\n";
        return 3;
    }

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
    std::cout << "eval samples: " << x.size() << "\n";
    std::vector<int> pred;
    if (model_type == "fgst") {
        FGSTConfig p_cfg;
        p_cfg.epochs = cfg.get_int("fgst.epochs", 50);
        p_cfg.batch_size = cfg.get_int("fgst.batch_size", 16);
        p_cfg.learning_rate = cfg.get_float("fgst.learning_rate", 1e-3f);
        p_cfg.weight_decay = cfg.get_float("fgst.weight_decay", 1e-4f);
        p_cfg.max_frames = cfg.get_int("fgst.max_frames", 32);
        p_cfg.max_points_per_frame =
            cfg.get_int("fgst.max_points_per_frame", 64);
        p_cfg.num_body_parts = cfg.get_int("fgst.num_body_parts", 4);
        p_cfg.point_feature_dim =
            cfg.get_int("fgst.point_feature_dim", 64);
        p_cfg.temporal_feature_dim =
            cfg.get_int("fgst.temporal_feature_dim", 128);
        FGSTModel model(p_cfg);
        if (!model.load(model_path)) {
            std::cerr << "failed to load fgst model: " << model_path << "\n";
            return 2;
        }
        pred = model.predict(seqs);
    } else if (model_type == "softmax") {
        SoftmaxClassifier model(SoftmaxConfig{});
        if (!load_model(model_path, model)) {
            std::cerr << "failed to load model: " << model_path << "\n";
            return 2;
        }
        pred = model.predict(x);
    } else {
        ClassifierConfig cls_cfg;
        cls_cfg.k_neighbors = 3;
        KNNClassifier model(cls_cfg);
        if (!load_model(model_path, model)) {
            std::cerr << "failed to load model: " << model_path << "\n";
            return 2;
        }
        pred = model.predict(x);
    }
    auto report = compute_metrics(y, pred);
    print_metrics(report);
    const std::string report_path = cfg.get("report.path", "./model/metrics_eval.csv");
    save_metrics_csv(report_path, report);
    std::cout << "metrics saved: " << report_path << "\n";
    return 0;
}

int run_predict(const std::string& config_path, const std::string& sample_csv_path) {
    Config cfg;
    if (!cfg.load(config_path)) {
        std::cerr << "failed to load config: " << config_path << "\n";
        return 1;
    }

    const std::string model_path = cfg.get("model.path", "model.bin");
    const std::string model_type = cfg.get("classifier.type", "knn");

    Sequence seq;
    if (!load_one_sequence_csv(sample_csv_path, seq)) {
        std::cerr << "failed to load sample csv: " << sample_csv_path << "\n";
        return 3;
    }

    Preprocessor preprocessor(read_preprocess(cfg));
    preprocessor.filter_sequence(seq);
    FeatureExtractor extractor;
    const float eps = cfg.get_float("cluster.eps", 0.25f);
    const std::size_t min_pts = static_cast<std::size_t>(cfg.get_int("cluster.min_pts", 5));

    const auto feat = extractor.extract(seq, preprocessor, eps, min_pts);
    int pred = -1;
    if (model_type == "fgst") {
        FGSTConfig p_cfg;
        p_cfg.epochs = cfg.get_int("fgst.epochs", 50);
        p_cfg.batch_size = cfg.get_int("fgst.batch_size", 16);
        p_cfg.learning_rate = cfg.get_float("fgst.learning_rate", 1e-3f);
        p_cfg.weight_decay = cfg.get_float("fgst.weight_decay", 1e-4f);
        p_cfg.max_frames = cfg.get_int("fgst.max_frames", 32);
        p_cfg.max_points_per_frame =
            cfg.get_int("fgst.max_points_per_frame", 64);
        p_cfg.num_body_parts = cfg.get_int("fgst.num_body_parts", 4);
        p_cfg.point_feature_dim =
            cfg.get_int("fgst.point_feature_dim", 64);
        p_cfg.temporal_feature_dim =
            cfg.get_int("fgst.temporal_feature_dim", 128);
        FGSTModel model(p_cfg);
        if (!model.load(model_path)) {
            std::cerr << "failed to load fgst model: " << model_path << "\n";
            return 2;
        }
        auto v = model.predict(std::vector<Sequence>{seq});
        pred = v.empty() ? -1 : v[0];
    } else if (model_type == "softmax") {
        SoftmaxClassifier model(SoftmaxConfig{});
        if (!load_model(model_path, model)) {
            std::cerr << "failed to load model: " << model_path << "\n";
            return 2;
        }
        pred = model.predict_one(feat);
    } else {
        ClassifierConfig cls_cfg;
        cls_cfg.k_neighbors = 3;
        KNNClassifier model(cls_cfg);
        if (!load_model(model_path, model)) {
            std::cerr << "failed to load model: " << model_path << "\n";
            return 2;
        }
        pred = model.predict_one(feat);
    }
    std::cout << "predict label: " << pred << "\n";
    const std::string predict_path = cfg.get("predict.output_path", "./model/predict_result.csv");
    std::ofstream ofs(predict_path);
    if (ofs.is_open()) {
        ofs << "sample_csv,predict_label\n";
        ofs << sample_csv_path << "," << pred << "\n";
        std::cout << "predict saved: " << predict_path << "\n";
    } else {
        std::cerr << "failed to write predict csv: " << predict_path << "\n";
    }
    return 0;
}

}  // namespace mmwave



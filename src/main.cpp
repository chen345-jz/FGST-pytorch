#include <iostream>
#include <string>

#include "pipeline.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n";
        std::cerr << "  mmwave_app train <config_path>\n";
        std::cerr << "  mmwave_app eval <config_path>\n";
        std::cerr << "  mmwave_app predict <config_path> <sample_npy_path>\n";
        return 1;
    }

    const std::string mode = argv[1];
    const std::string config_path = argv[2];

    if (mode == "train") {
        return mmwave::run_train(config_path);
    }
    if (mode == "eval") {
        return mmwave::run_eval(config_path);
    }
    if (mode == "predict") {
        if (argc < 4) {
            std::cerr << "predict mode requires sample_npy_path\n";
            return 1;
        }
        return mmwave::run_predict(config_path, argv[3]);
    }

    std::cerr << "unknown mode: " << mode << "\n";
    return 1;
}

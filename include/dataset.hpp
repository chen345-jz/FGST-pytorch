#pragma once

#include <string>

#include "types.hpp"

namespace mmwave {

class DatasetLoader {
public:
    // Expected folder layout:
    // root/p_1/*.npy, root/p_2/*.npy, ...
    // Each npy is shape (T, P, 4): x,y,z,doppler
    bool load_from_npy_root(const std::string& root_path, Dataset& out_dataset) const;
};

SplitData split_dataset(const Dataset& dataset, float test_ratio, unsigned int seed);

}  // namespace mmwave

#pragma once

#include <string>

#include "types.hpp"

namespace mmwave {

class DatasetLoader {
public:
    // Expected manifest CSV columns:
    // sequence_id,label,file_path
    // Each sequence file columns:
    // frame_idx,timestamp,x,y,z,doppler,snr
    bool load_from_manifest(const std::string& manifest_path, Dataset& out_dataset) const;
};

SplitData split_dataset(const Dataset& dataset, float test_ratio, unsigned int seed);

}  // namespace mmwave

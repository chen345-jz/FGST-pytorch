#pragma once

#include "preprocess.hpp"
#include "types.hpp"

namespace mmwave {

class FeatureExtractor {
public:
    FeatureVector extract(const Sequence& sequence, const Preprocessor& preprocessor,
                          float cluster_eps, std::size_t cluster_min_pts) const;
};

}  // namespace mmwave

#pragma once

#include <string>

#include "classifier.hpp"
#include "softmax_classifier.hpp"

namespace mmwave {

bool save_model(const std::string& path, const KNNClassifier& model);
bool load_model(const std::string& path, KNNClassifier& model);
bool save_model(const std::string& path, const SoftmaxClassifier& model);
bool load_model(const std::string& path, SoftmaxClassifier& model);

}  // namespace mmwave

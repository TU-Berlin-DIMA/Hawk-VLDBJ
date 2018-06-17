#include "Configuration.hpp"
#include "Feature.hpp"

namespace perseus {

Feature* Configuration::getFeature(const std::string& name) const {
  // iterator and string comparison => slow?
  // an unordered_map would give constant average performance
  // but then we have to maintain two data structures
  for (auto& feature : features_) {
    if (feature->name() == name) {
      return feature.get();
    }
  }
  return nullptr;
}

void Configuration::addFeature(std::unique_ptr<Feature> feature) {
  std::string name = feature->name();
  for (auto& existingFeature : features_) {
    if (existingFeature->name() == name) {
      throw std::logic_error("Feature with same name already exists: " + name);
    }
  }
  features_.push_back(std::move(feature));
}

const std::vector<Feature*> Configuration::features() const {
  return convertUniquePtrElementsToRawPointers(features_);
}

const bool Configuration::operator==(const Configuration& other) const {
  return features_ == other.features_;
}

std::ostream& operator<<(std::ostream& ostream,
                         const Configuration& configuration) {
  ostream << "Configuration(";
  auto features = configuration.features();
  for (auto iterator = features.begin(); iterator != features.end();
       ++iterator) {
    if (iterator != features.begin()) {
      ostream << ", ";
    }
    auto feature = *iterator;
    ostream << feature->toString();
  }
  ostream << ")";
  return ostream;
}
}
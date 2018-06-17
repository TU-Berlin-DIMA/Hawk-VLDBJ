#ifndef PERSEUS_CONFIGURATION_HPP
#define PERSEUS_CONFIGURATION_HPP

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "DiscreteFeature.hpp"
#include "utils.h"

namespace perseus {

  class Feature;

  class Configuration {
   private:
    std::vector<std::unique_ptr<Feature>> features_;

   public:
    // assignment operator needed to erase from vectors
    void operator=(const Configuration& other) {
      features_.clear();
      for (auto feature : other.features()) {
        features_.push_back(feature->clone());
      }
    }
    // copy constructor needed for testing
    Configuration(const Configuration& other) { operator=(other); }
    // default constructor needed
    Configuration() = default;

    void addFeature(std::unique_ptr<Feature> feature);

    Feature* getFeature(const std::string& name) const;

    template <typename T>
    DiscreteFeature<T>* getTypedFeature(const std::string& name) const {
      return dynamic_cast<DiscreteFeature<T>*>(getFeature(name));
    }

    template <typename T>
    T getValue(const std::string& name) const {
      DiscreteFeature<T>* feature = getTypedFeature<T>(name);
      if (feature) {
        return feature->value();
      } else {
        throw std::logic_error("Tried to access value of unknown feature: " +
                               name);
      }
    }

    bool nextConfiguration() {
      for (auto& feature : features_) {
        if (feature->nextValue()) {
          return true;
        }
      }
      return false;
    }

    const std::vector<Feature*> features() const;

    const bool operator==(const Configuration& other) const;
  };

  std::ostream& operator<<(std::ostream& ostream,
                           const Configuration& configuration);
}

#endif  // PERSEUS_CONFIGURATION_HPP

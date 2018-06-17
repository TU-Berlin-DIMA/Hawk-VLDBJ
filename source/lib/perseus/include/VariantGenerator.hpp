#ifndef PERSEUS_VARIANTGENERATOR_HPP
#define PERSEUS_VARIANTGENERATOR_HPP

#include <memory>
#include <string>
#include <vector>
#include "Feature.hpp"
#include "Variant.hpp"

namespace perseus {

  class Configuration;

  class VariantGenerator {
   public:
    virtual std::unique_ptr<Variant> createVariant(
        std::unique_ptr<Configuration> configuration) const = 0;

    virtual const std::vector<Feature*> features() const = 0;

    virtual ~VariantGenerator() {}

    virtual const std::string name() const = 0;

    virtual const bool validateConfiguration(
        const Configuration& configuration) const = 0;
  };
}

#endif  // PERSEUS_VARIANTGENERATOR_HPP

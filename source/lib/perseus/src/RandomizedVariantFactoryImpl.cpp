#include "RandomizedVariantFactoryImpl.hpp"
#include "Configuration.hpp"
#include "DiscreteFeature.hpp"

namespace perseus {

std::unique_ptr<Variant> RandomizedVariantFactoryImpl::createRandomVariant(
    const VariantGenerator& generator, const size_t chunkSize) const {
  auto features = generator.features();
  auto chunkSizeFeature = std::unique_ptr<Feature>(
      new DiscreteFeature<size_t>("chunk_size", {chunkSize}));
  features.push_back(chunkSizeFeature.get());
  auto configuration = std::unique_ptr<Configuration>(new Configuration);
  for (auto feature : features) {
    auto copy = feature->clone();
    configuration->addFeature(std::move(copy));
  }
  bool valid = false;
  while (!valid) {
    for (auto feature : configuration->features()) {
      feature->randomize();
    }
    valid = generator.validateConfiguration(*configuration);
  }
  return generator.createVariant(std::move(configuration));
}
}
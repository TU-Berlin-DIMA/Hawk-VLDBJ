#include "RandomizedInitializationStrategy.hpp"
#include "Configuration.hpp"
#include "Feature.hpp"
#include "VariantGenerator.hpp"

namespace perseus {

std::unique_ptr<Configuration>
RandomizedInitializationStrategy::nextConfiguration(
    const std::vector<Feature*>& features,
    const VariantGenerator& generator) const {
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
  return configuration;
}

void RandomizedInitializationStrategy::initializePool(VariantPool& pool) {
  pool.reset();
  auto& generator = pool.generator();
  while (pool.poolSize() < pool.initialSize()) {
    auto features = generator.features();
    auto chunkSizeFeature = std::unique_ptr<Feature>(
        new DiscreteFeature<size_t>("chunk_size", {pool.chunkSize()}));
    features.push_back(chunkSizeFeature.get());
    auto configuration = nextConfiguration(features, generator);
    auto variant = generator.createVariant(std::move(configuration));
    pool.addVariant(std::move(variant));
  }
}
}

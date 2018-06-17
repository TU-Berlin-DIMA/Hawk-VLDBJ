#include "VWGreedyBaselineStrategy.hpp"
#include "Configuration.hpp"
#include "DiscreteFeature.hpp"
#include "VariantGenerator.hpp"

namespace perseus {

void VWGreedyBaselineStrategy::initializePool(VariantPool& pool) {
  pool.reset();
  auto& generator = pool.generator();
  auto features = generator.features();
  Configuration configuration;
  configuration.addFeature(std::unique_ptr<Feature>(
      new DiscreteFeature<size_t>("chunk_size", {pool.chunkSize()})));
  for (auto feature : features) {
    configuration.addFeature(std::unique_ptr<Feature>(feature->clone()));
  }
  do {
    if (generator.validateConfiguration(configuration)) {
      auto variant = generator.createVariant(
          std::unique_ptr<Configuration>(new Configuration(configuration)));
      pool.addVariant(std::move(variant));
    }
  } while (configuration.nextConfiguration());
}
}
#include "FullPoolBaselineStrategy.hpp"
#include "Configuration.hpp"
#include "DiscreteFeature.hpp"
#include "VariantGenerator.hpp"
#include "utils.h"

namespace perseus {

void FullPoolBaselineStrategy::initializePool(VariantPool& pool) {
  pool.reset();
  auto& generator = pool.generator();
  auto configurations = createConfigurations(generator, pool.chunkSize());
  remaining_ = configurations.size() - elitism();
  for (auto& configuration : configurations) {
    auto variant = generator.createVariant(
        std::unique_ptr<Configuration>(new Configuration(configuration)));
    pool.addVariant(std::move(variant));
  }
}

std::vector<std::unique_ptr<Variant>>
FullPoolBaselineStrategy::createNewVariants(
    const VariantPool& pool, size_t count,
    const std::vector<std::tuple<double, unsigned>> variantScores) {
  remaining_ = std::max(0, (signed)(remaining_ - count));
  PERSEUS_TRACE {
    if (remaining_ == 0) {
      BOOST_LOG_TRIVIAL(trace) << "full-pool: Configurations exhausted.";
    } else {
      BOOST_LOG_TRIVIAL(trace) << "full-pool: Remaining configurations: "
                               << remaining_;
    }
  };
  // no op
  return {};
}

const std::vector<Configuration> FullPoolBaselineStrategy::createConfigurations(
    const VariantGenerator& generator, const size_t chunk_size) const {
  std::vector<Configuration> configurations;
  auto features = generator.features();
  Configuration configuration;
  configuration.addFeature(std::unique_ptr<Feature>(
      new DiscreteFeature<size_t>("chunk_size", {chunk_size})));
  for (auto feature : features) {
    configuration.addFeature(std::unique_ptr<Feature>(feature->clone()));
  }
  do {
    if (generator.validateConfiguration(configuration)) {
      configurations.push_back(configuration);
    }
  } while (configuration.nextConfiguration());
  std::shuffle(configurations.begin(), configurations.end(), global_rnd);
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "full-pool: Configurations at start: "
                             << configurations.size();
  };
  return configurations;
}
}
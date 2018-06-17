#include "RandomizedUpdateStrategy.hpp"
#include <iostream>
#include "Configuration.hpp"
#include "utils.h"

namespace perseus {

std::vector<std::unique_ptr<Variant>>
RandomizedUpdateStrategy::createNewVariants(
    const VariantPool& pool, size_t count,
    const std::vector<std::tuple<double, unsigned>> variantScores) {
  std::vector<std::unique_ptr<Variant>> variants;
  auto& generator = pool.generator();
  for (auto i = 0u; i < count; ++i) {
    auto variant = factory_->createRandomVariant(generator, pool.chunkSize());
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Creating new variant: " << *variant;
    };
    variants.push_back(std::move(variant));
  }
  return variants;
}
}

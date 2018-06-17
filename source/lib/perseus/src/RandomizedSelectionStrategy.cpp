#include "RandomizedSelectionStrategy.hpp"
#include <stdexcept>
#include "VariantPool.hpp"

#include <stdexcept>

namespace perseus {

extern std::mt19937 global_rnd;

void RandomizedSelectionStrategy::reset(const VariantPool& variantPool,
                                        const bool clearPerformance) {
  pool = &variantPool;
  unsigned int poolSize = pool->poolSize();
  distribution_ = std::uniform_int_distribution<unsigned int>(0, poolSize - 1);
}

Variant& RandomizedSelectionStrategy::selectVariant() {
  if (!pool) {
    throw std::logic_error("Randomized selection strategy is not initialized");
  }
  unsigned int index = distribution_(global_rnd);
  return pool->getVariant(index);
}
}
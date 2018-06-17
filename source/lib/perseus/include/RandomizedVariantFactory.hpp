#ifndef PERSEUS_RANDOMIZEDVARIANTFACTORY_HPP
#define PERSEUS_RANDOMIZEDVARIANTFACTORY_HPP

#include <memory>
#include "Variant.hpp"
#include "VariantGenerator.hpp"

namespace perseus {

  class RandomizedVariantFactory {
   public:
    virtual std::unique_ptr<Variant> createRandomVariant(
        const VariantGenerator& generator, const size_t chunkSize) const = 0;
    virtual ~RandomizedVariantFactory() {}
  };
}

#endif  // PERSEUS_RANDOMIZEDVARIANTFACTORY_HPP

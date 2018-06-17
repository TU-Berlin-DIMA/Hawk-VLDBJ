#ifndef PERSEUS_RANDOMIZEDVARIANTFACTORYIMPL_HPP
#define PERSEUS_RANDOMIZEDVARIANTFACTORYIMPL_HPP

#include "RandomizedVariantFactory.hpp"

namespace perseus {

  class RandomizedVariantFactoryImpl : public RandomizedVariantFactory {
   public:
    virtual std::unique_ptr<Variant> createRandomVariant(
        const VariantGenerator& generator, const size_t chunkSize) const;
  };
}

#endif  // PERSEUS_RANDOMIZEDVARIANTFACTORYIMPL_HPP

#ifndef PERSEUS_INITIALIZATIONSTRATEGY_HPP
#define PERSEUS_INITIALIZATIONSTRATEGY_HPP

#include <memory>
#include <vector>
#include "VariantPool.hpp"

namespace perseus {

  class VariantGenerator;

  class Configuration;

  class Feature;

  class InitializationStrategy {
   public:
    virtual void initializePool(VariantPool& pool) = 0;

    virtual ~InitializationStrategy() {}
  };
}

#endif  // PERSEUS_INITIALIZATIONSTRATEGY_HPP

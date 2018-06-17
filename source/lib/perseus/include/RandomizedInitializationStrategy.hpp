#ifndef PERSEUS_RANDOMIZEDINITIALIZESTRATEGY_HPP
#define PERSEUS_RANDOMIZEDINITIALIZESTRATEGY_HPP

#include <memory>
#include "InitializationStrategy.hpp"

namespace perseus {

  class RandomizedInitializationStrategy : public InitializationStrategy {
   public:
    virtual void initializePool(VariantPool& pool) override;

   public:
    // expose method for testing although I'm not quite sure I need it
    virtual std::unique_ptr<Configuration> nextConfiguration(
        const std::vector<Feature*>& features,
        const VariantGenerator& generator) const;
  };
}

#endif  // PERSEUS_RANDOMIZEDINITIALIZESTRATEGY_HPP

#ifndef PERSEUS_VWGREEDYBASELINESTRATEGY_HPP
#define PERSEUS_VWGREEDYBASELINESTRATEGY_HPP

#include <memory>
#include "InitializationStrategy.hpp"
#include "UpdateStrategy.hpp"

namespace perseus {

  class VWGreedyBaselineStrategy : public InitializationStrategy,
                                   public UpdateStrategy {
   public:
    virtual void initializePool(VariantPool& pool) override;

    virtual void updatePool(VariantPool& pool) override {
      // no op
    }

    virtual void reset() override {}
  };
}

#endif  // PERSEUS_VWGREEDYBASELINESTRATEGY_HPP

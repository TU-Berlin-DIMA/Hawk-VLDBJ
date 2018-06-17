#ifndef PERSEUS_RANDOMIZEDSELECTIONSTRATEGY_HPP
#define PERSEUS_RANDOMIZEDSELECTIONSTRATEGY_HPP

#include <random>
#include "SelectionStrategy.hpp"

namespace perseus {

  class RandomizedSelectionStrategy : public SelectionStrategy {
   private:
    const VariantPool* pool = nullptr;
    std::uniform_int_distribution<unsigned int> distribution_;

   public:
    virtual Variant& selectVariant() override;
    virtual void reset(const VariantPool& variantPool,
                       const bool clearPerformance = true) override;
    virtual void finishQuery() override {}
  };
}

#endif  // PERSEUS_RANDOMIZEDSELECTIONSTRATEGY_HPP

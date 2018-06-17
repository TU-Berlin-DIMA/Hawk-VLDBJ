#ifndef PERSEUS_UPDATESTRATEGY_HPP
#define PERSEUS_UPDATESTRATEGY_HPP

#include "VariantPool.hpp"

namespace perseus {

  class UpdateStrategy {
   public:
    virtual void updatePool(VariantPool& pool) = 0;
    virtual void reset() = 0;
    virtual ~UpdateStrategy() {}
  };
}

#endif  // PERSEUS_UPDATESTRATEGY_HPP

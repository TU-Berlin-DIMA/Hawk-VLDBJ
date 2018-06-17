#ifndef PERSEUS_TOTALRUNTIMEVARIANTSCORER_HPP
#define PERSEUS_TOTALRUNTIMEVARIANTSCORER_HPP

#include "ErrorFunctionVariantScorer.hpp"

namespace perseus {

  class TotalRuntimeVariantScorer : public ErrorFunctionVariantScorer {
   public:
    TotalRuntimeVariantScorer() : ErrorFunctionVariantScorer("Total runtime") {}

   protected:
    virtual double getMetric(const Variant& variant) const override;
  };
}

#endif  // PERSEUS_TOTALRUNTIMEVARIANTSCORER_HPP

#ifndef PERSEUS_CURRENTRUNTIMEVARIANTSCORER_HPP
#define PERSEUS_CURRENTRUNTIMEVARIANTSCORER_HPP

#include "ErrorFunctionVariantScorer.hpp"

namespace perseus {

  class CurrentRuntimeVariantScorer : public ErrorFunctionVariantScorer {
   public:
    CurrentRuntimeVariantScorer()
        : ErrorFunctionVariantScorer("Current runtime") {}

   protected:
    virtual double getMetric(const Variant& variant) const override;
  };
}

#endif  // PERSEUS_CURRENTRUNTIMEVARIANTSCORER_HPP

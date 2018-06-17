#include "CurrentRuntimeVariantScorer.hpp"
#include "Variant.hpp"

namespace perseus {

double CurrentRuntimeVariantScorer::getMetric(const Variant& variant) const {
  return variant.currentRuntimePerTuple();
  ;
}
}
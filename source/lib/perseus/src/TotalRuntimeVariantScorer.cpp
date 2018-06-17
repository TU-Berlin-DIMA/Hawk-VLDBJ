#include "TotalRuntimeVariantScorer.hpp"
#include "Variant.hpp"

namespace perseus {

double TotalRuntimeVariantScorer::getMetric(const Variant& variant) const {
  return (double)variant.totalRuntime() / variant.totalTuples();
}
}
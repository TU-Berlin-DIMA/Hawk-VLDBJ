#include "TotalCallsVariantScorer.hpp"
#include <tuple>
#include "Variant.hpp"

namespace perseus {

const std::vector<std::tuple<double, unsigned>>
TotalCallsVariantScorer::scoreVariants(std::vector<Variant*> variants) const {
  auto totalCalls = 0ul;
  for (auto variant : variants) {
    totalCalls += variant->totalCalls();
  }
  std::vector<std::tuple<double, unsigned>> fitness;
  for (auto i = 0u; i < variants.size(); ++i) {
    auto variant = variants[i];
    auto relativeCalls =
        variant->totalCalls() / static_cast<double>(totalCalls);
    fitness.push_back(std::make_tuple(relativeCalls, i));
  }
  return fitness;
}
}

#include "ErrorFunctionVariantScorer.hpp"
#include "Variant.hpp"
#include "utils.h"

namespace perseus {

const std::vector<std::tuple<double, unsigned>>
ErrorFunctionVariantScorer::scoreVariants(
    std::vector<Variant*> variants) const {
  auto mean = 0.0;
  auto squaredDeviation = 0.0;
  auto tested = 0u;
  for (auto i = 0u; i < variants.size(); ++i) {
    auto variant = variants[i];
    auto runtime = getMetric(*variant);
    if (runtime > 0.0) {
      tested += 1;
      double delta = runtime - mean;
      mean += delta / (i + 1);
      squaredDeviation += delta * (runtime - mean);
    }
  }
  auto variance = squaredDeviation / (tested - 1);
  auto totalProbability = 0.0;
  std::vector<std::tuple<double, unsigned>> scores(tested);
  for (auto i = 0u; i < variants.size(); ++i) {
    auto variant = variants[i];
    auto runtime = getMetric(*variant);
    if (runtime > 0.0) {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << description_ << " of variant " << i << ", "
                                 << *variant << ": " << runtime;
      };
      auto probability =
          1.0 - 0.5 * (1.0 + erf((runtime - mean) / sqrt(2 * variance)));
      totalProbability += probability;
      scores[i] = std::make_tuple(probability, i);
    } else {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Keeping untested variant " << i;
      };
    }
  }
  for (auto& score : scores) {
    std::get<0>(score) /= totalProbability;
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Score of variant " << std::get<1>(score)
                               << ": " << std::get<0>(score);
    };
  }
  return scores;
}
}
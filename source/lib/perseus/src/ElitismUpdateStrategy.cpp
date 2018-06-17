#include "ElitismUpdateStrategy.hpp"
#include <algorithm>
#include "utils.h"

namespace perseus {

void ElitismUpdateStrategy::updatePool(VariantPool& pool) {
  auto scores = variantScorer_->scoreVariants(pool.variants());
  std::sort(scores.begin(), scores.end(),
            std::greater<std::tuple<double, unsigned>>());
  auto newVariants = createNewVariants(
      pool, computeNumberOfNewVariants(pool, scores.size()), scores);
  removeSlowVariants(pool, scores);
  addNewVariants(pool, std::move(newVariants));
}

const unsigned ElitismUpdateStrategy::computeNumberOfNewVariants(
    const VariantPool& pool, const size_t numberOfTestedVariants) const {
  auto newPoolSize = std::max(
      static_cast<size_t>(std::round(pool.poolSize() * pool.reductionFactor())),
      pool.minimumSize());
  auto numberOfUntestedVariants = pool.poolSize() - numberOfTestedVariants;
  auto maxNewVariants =
      static_cast<signed>(newPoolSize - elitism_ - numberOfUntestedVariants);
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace)
        << "pool.poolSize() = " << pool.poolSize()
        << "; pool.reductionFactor() = " << pool.reductionFactor()
        << "; pool.minimumSize() = " << pool.minimumSize()
        << "; newPoolSize = " << newPoolSize
        << "; numberOfTestedVariants = " << numberOfTestedVariants
        << "; elitism_ = " << elitism_
        << "; maxNewVariants = " << maxNewVariants;
  };
  return std::max(maxNewVariants, 0);
}

void ElitismUpdateStrategy::removeSlowVariants(
    VariantPool& pool,
    const std::vector<std::tuple<double, unsigned>> variantScores) const {
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "Keeping " << elitism_ << " fastest variants.";
  };
  // remove slow variants in order
  std::vector<unsigned> toRemove;
  for (auto i = elitism_; i < variantScores.size(); ++i) {
    auto index = std::get<1>(variantScores[i]);
    toRemove.push_back(index);
  }
  std::sort(toRemove.begin(), toRemove.end(), std::greater<unsigned>());
  for (auto i : toRemove) {
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Removing slow variant: " << i;
    };
    pool.removeVariant(i);
  }
}

void ElitismUpdateStrategy::addNewVariants(
    VariantPool& pool, std::vector<std::unique_ptr<Variant>> variants) const {
  for (auto& variant : variants) {
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Adding variant to pool: " << *variant;
    };
    pool.addVariant(std::move(variant));
  }
}
}
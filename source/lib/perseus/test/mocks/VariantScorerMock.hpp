#ifndef PERSEUS_VARIANTSCORERMOCK_HPP
#define PERSEUS_VARIANTSCORERMOCK_HPP

#include <gmock/gmock.h>
#include <vector>
#include "VariantScorer.hpp"

namespace perseus {

  class Variant;

  class VariantScorerMock : public VariantScorer {
   public:
    MOCK_CONST_METHOD1(
        scoreVariants,
        const std::vector<std::tuple<double, unsigned>>(std::vector<Variant*>));
  };
}

#endif  // PERSEUS_VARIANTSCORERMOCK_HPP

#ifndef PERSEUS_TOTALCALLSVARIANTSCORER_HPP
#define PERSEUS_TOTALCALLSVARIANTSCORER_HPP

#include "VariantScorer.hpp"

namespace perseus {

  class TotalCallsVariantScorer : public VariantScorer {
   public:
    const std::vector<std::tuple<double, unsigned>> scoreVariants(
        std::vector<Variant*> variants) const override;
  };
}

#endif  // PERSEUS_TOTALCALLSVARIANTSCORER_HPP

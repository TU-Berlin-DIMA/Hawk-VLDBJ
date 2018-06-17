#ifndef PERSEUS_VARIANTSCORER_HPP
#define PERSEUS_VARIANTSCORER_HPP

#include <utility>
#include <vector>

namespace perseus {

  class Variant;

  class VariantScorer {
   public:
    virtual const std::vector<std::tuple<double, unsigned>> scoreVariants(
        std::vector<Variant*> variants) const = 0;
  };
}

#endif  // PERSEUS_VARIANTSCORER_HPP

#ifndef PERSEUS_ERRORFUNCTIONVARIANTSCORER_HPP
#define PERSEUS_ERRORFUNCTIONVARIANTSCORER_HPP

#include <string>
#include "VariantScorer.hpp"

namespace perseus {

  class ErrorFunctionVariantScorer : public VariantScorer {
   private:
    const std::string description_;

   public:
    virtual const std::vector<std::tuple<double, unsigned>> scoreVariants(
        std::vector<Variant*> variants) const override;

   protected:
    ErrorFunctionVariantScorer(const std::string& description)
        : description_(description) {}

    virtual double getMetric(const Variant& variant) const = 0;
  };
}

#endif  // PERSEUS_ERRORFUNCTIONVARIANTSCORER_HPP

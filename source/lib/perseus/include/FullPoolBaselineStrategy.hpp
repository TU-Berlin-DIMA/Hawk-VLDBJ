#ifndef PERSEUS_FULLPOOLBASELINESTRATEGY_HPP
#define PERSEUS_FULLPOOLBASELINESTRATEGY_HPP

#include <memory>
#include <vector>
#include "ElitismUpdateStrategy.hpp"
#include "InitializationStrategy.hpp"

namespace perseus {

  class FullPoolBaselineStrategy : public InitializationStrategy,
                                   public ElitismUpdateStrategy {
    using ElitismUpdateStrategy::ElitismUpdateStrategy;

   private:
    unsigned remaining_ = 0;

   public:
    virtual void initializePool(VariantPool& pool) override;

   private:
    const std::vector<Configuration> createConfigurations(
        const VariantGenerator& generator, const size_t chunk_size) const;

    virtual std::vector<std::unique_ptr<Variant>> createNewVariants(
        const VariantPool& pool, size_t count,
        const std::vector<std::tuple<double, unsigned>> variantScores) override;
  };
}

#endif  // PERSEUS_FULLPOOLBASELINESTRATEGY_HPP

#ifndef PERSEUS_MARKOVUPDATESTRATEGY_HPP
#define PERSEUS_MARKOVUPDATESTRATEGY_HPP

#include <random>
#include <tuple>
#include "Configuration.hpp"
#include "UpdateStrategy.hpp"
#include "VariantGenerator.hpp"
#include "VariantScorer.hpp"

namespace perseus {

  class MarkovUpdateStrategy : public UpdateStrategy {
   private:
    const std::shared_ptr<VariantScorer> variantScorer_;
    std::uniform_real_distribution<double> selectionDistribution;

   public:
    MarkovUpdateStrategy(std::shared_ptr<VariantScorer> variantScorer)
        : variantScorer_(variantScorer) {}

    virtual void updatePool(VariantPool& pool);

    virtual void reset() {}

   private:
    void determineVariantsToKeep(
        VariantPool& pool,
        const std::vector<std::tuple<double, unsigned>>& scores);

    void createNewSuccessors(VariantPool& pool, const size_t chains);

    std::unique_ptr<Configuration> createSuccessor(const VariantPool& pool,
                                                   const unsigned chain) const;

    void removeChains(VariantPool& pool, const size_t count,
                      const std::vector<std::tuple<double, unsigned>>& scores);

    const size_t computeNewPoolSize(const VariantPool& pool) const;

    const size_t computeChainsToKeep(const size_t newPoolSize,
                                     const size_t testedVariants) const;

    const size_t computeChainsToRemove(const size_t poolSize,
                                       const size_t newPoolSize) const;
  };
}

#endif  // PERSEUS_MARKOVUPDATESTRATEGY_HPP

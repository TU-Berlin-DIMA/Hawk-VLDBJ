#ifndef PERSEUS_GENETICUPDATESTRATEGY_HPP
#define PERSEUS_GENETICUPDATESTRATEGY_HPP

#include <random>
#include "ElitismUpdateStrategy.hpp"
#include "utils.h"

namespace perseus {

  class GeneticUpdateStrategy : public ElitismUpdateStrategy {
   private:
    const double mutationProbability_;
    const double matingProbability_ = 0.5;

   public:
    GeneticUpdateStrategy(std::shared_ptr<VariantScorer> variantScorer,
                          const unsigned elitism,
                          const double mutationProbability)
        : ElitismUpdateStrategy(variantScorer, elitism),
          mutationProbability_(mutationProbability) {}

   private:
    std::uniform_real_distribution<double> matingDistribution_;

    virtual std::vector<std::unique_ptr<Variant>> createNewVariants(
        const VariantPool& pool, size_t count,
        const std::vector<std::tuple<double, unsigned>> variantScores) override;

    const std::vector<std::tuple<double, unsigned>> computeMatingProbabilites(
        const std::vector<std::tuple<double, unsigned>>& variantScores) const;

    const unsigned findMate(
        const std::vector<std::tuple<double, unsigned>> matingProbabilities,
        const unsigned exclude = -1);

    std::unique_ptr<Configuration> createOffspring(const VariantPool& pool,
                                                   unsigned mate1,
                                                   unsigned mate2);
  };
}

#endif  // PERSEUS_GENETICUPDATESTRATEGY_HPP

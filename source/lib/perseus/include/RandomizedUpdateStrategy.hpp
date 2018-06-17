#ifndef PERSEUS_RANDOMIZEDUPDATESTRATEGY_HPP
#define PERSEUS_RANDOMIZEDUPDATESTRATEGY_HPP

#include <memory>
#include "ElitismUpdateStrategy.hpp"
#include "RandomizedVariantFactoryImpl.hpp"
#include "VariantGenerator.hpp"

namespace perseus {

  class RandomizedUpdateStrategy : public ElitismUpdateStrategy {
   private:
    std::unique_ptr<RandomizedVariantFactory> factory_;

   public:
    RandomizedUpdateStrategy(std::shared_ptr<VariantScorer> variantScorer,
                             const unsigned elitism)
        : ElitismUpdateStrategy(variantScorer, elitism),
          factory_(std::unique_ptr<RandomizedVariantFactory>(
              new RandomizedVariantFactoryImpl)) {}

    // exposed for testing
    void setRandomizedVariantFactory(
        std::unique_ptr<RandomizedVariantFactory> factory) {
      factory_ = std::move(factory);
    }

   private:
    virtual std::vector<std::unique_ptr<Variant>> createNewVariants(
        const VariantPool& pool, size_t count,
        const std::vector<std::tuple<double, unsigned>> variantScores) override;
  };
}

#endif  // PERSEUS_RANDOMIZEDUPDATESTRATEGY_HPP

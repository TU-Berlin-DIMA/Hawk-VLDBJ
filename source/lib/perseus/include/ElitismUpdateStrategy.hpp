#ifndef PERSEUS_ELITISMUPDATESTRATEGY_HPP
#define PERSEUS_ELITISMUPDATESTRATEGY_HPP

#include <memory>
#include "UpdateStrategy.hpp"
#include "VariantScorer.hpp"

namespace perseus {

  class ElitismUpdateStrategy : public UpdateStrategy {
   private:
    const std::shared_ptr<VariantScorer> variantScorer_;
    const unsigned elitism_;

   public:
    ElitismUpdateStrategy(std::shared_ptr<VariantScorer> variantScorer,
                          const unsigned elitism)
        : variantScorer_(variantScorer), elitism_(elitism) {}

    virtual void updatePool(VariantPool& pool) override;

    virtual void reset() override {}

    const unsigned elitism() const { return elitism_; }

   private:
    void removeSlowVariants(
        VariantPool& pool,
        const std::vector<std::tuple<double, unsigned>> variantScores) const;

    void addNewVariants(VariantPool& pool,
                        std::vector<std::unique_ptr<Variant>> variants) const;

    const unsigned computeNumberOfNewVariants(
        const VariantPool& pool, const size_t numberOfTestedVariants) const;

    virtual std::vector<std::unique_ptr<Variant>> createNewVariants(
        const VariantPool& pool, size_t count,
        const std::vector<std::tuple<double, unsigned>> variantScores) = 0;
  };
}

#endif  // PERSEUS_ELITISMUPDATESTRATEGY_HPP

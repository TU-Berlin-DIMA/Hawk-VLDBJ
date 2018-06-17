#ifndef PERSEUS_DEFAULTVARIANTPOOL_HPP
#define PERSEUS_DEFAULTVARIANTPOOL_HPP

#include <memory>
#include <vector>
#include "InitializationStrategy.hpp"
#include "SelectionStrategy.hpp"
#include "UpdateStrategy.hpp"
#include "VariantGenerator.hpp"
#include "VariantPool.hpp"

namespace perseus {

  class DefaultVariantPool : public VariantPool {
   private:
    const size_t initialSize_;
    const double reductionFactor_;
    const size_t minimumSize_;
    const size_t chunkSize_;
    std::unique_ptr<VariantGenerator> generator_;
    std::shared_ptr<InitializationStrategy> initializationStrategy_;
    std::unique_ptr<SelectionStrategy> selectionStrategy_;
    std::shared_ptr<UpdateStrategy> updateStrategy_;
    std::vector<std::unique_ptr<Variant>> variants_;

   public:
    DefaultVariantPool(
        const size_t poolSize, const size_t chunkSize,
        std::unique_ptr<VariantGenerator> generator,
        std::shared_ptr<InitializationStrategy> initializationStrategy,
        std::unique_ptr<SelectionStrategy> selectionStrategy,
        std::shared_ptr<UpdateStrategy> updateStrategy)
        : DefaultVariantPool(poolSize, 1.0, poolSize, chunkSize,
                             std::move(generator), initializationStrategy,
                             std::move(selectionStrategy), updateStrategy) {}

    DefaultVariantPool(
        const size_t initialSize, const double reductionFactor,
        const size_t minimumSize, const size_t chunkSize,
        std::unique_ptr<VariantGenerator> generator,
        std::shared_ptr<InitializationStrategy> initializationStrategy,
        std::unique_ptr<SelectionStrategy> selectionStrategy,
        std::shared_ptr<UpdateStrategy> updateStrategy);

    virtual const size_t poolSize() const override { return variants_.size(); }

    virtual const size_t initialSize() const override { return initialSize_; }

    virtual const std::vector<Variant*> variants() const override;

    virtual const VariantGenerator& generator() const override {
      return *generator_;
    }

    virtual Variant& getVariant() override;

    virtual Variant& getVariant(unsigned index) const override {
      return *variants_.at(index);
    }

    virtual void removeVariant(unsigned index) override {
      variants_.erase(variants_.begin() + index);
    }

    virtual void addVariant(std::unique_ptr<Variant> variant) override {
      variants_.push_back(std::move(variant));
    }

    virtual void swapVariants(const unsigned i, const unsigned j) override;

    virtual void updateVariant(const unsigned index,
                               std::unique_ptr<Variant> variant) override;

    virtual const std::string name() const override;

    virtual const size_t chunkSize() const override { return chunkSize_; }

    virtual void finishQuery() override;

    virtual void update() override;

    virtual const double reductionFactor() const override {
      return reductionFactor_;
    }

    virtual const size_t minimumSize() const override { return minimumSize_; }

    virtual void initialize() override;

    virtual void reset() override;
  };
}

#endif  // PERSEUS_DEFAULTVARIANTPOOL_HPP

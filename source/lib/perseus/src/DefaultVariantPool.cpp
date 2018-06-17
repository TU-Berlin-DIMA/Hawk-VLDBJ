#include "DefaultVariantPool.hpp"
#include <cassert>
#include <utility>
#include "Configuration.hpp"
#include "DiscreteFeature.hpp"
#include "InitializationStrategy.hpp"
#include "SelectionStrategy.hpp"
#include "VariantGenerator.hpp"
#include "VariantPool.hpp"

namespace perseus {
DefaultVariantPool::DefaultVariantPool(
    const size_t initialSize, const double reductionFactor,
    const size_t minimumSize, const size_t chunkSize,
    std::unique_ptr<VariantGenerator> generator,
    std::shared_ptr<InitializationStrategy> initializationStrategy,
    std::unique_ptr<SelectionStrategy> selectionStrategy,
    std::shared_ptr<UpdateStrategy> updateStrategy)
    : initialSize_(initialSize),
      reductionFactor_(reductionFactor),
      minimumSize_(minimumSize),
      chunkSize_(chunkSize),
      generator_(std::move(generator)),
      initializationStrategy_(initializationStrategy),
      selectionStrategy_(std::move(selectionStrategy)),
      updateStrategy_(updateStrategy) {
  assert(initialSize_ >= minimumSize_);
  assert(reductionFactor_ <= 1.0);
}

const std::vector<Variant*> DefaultVariantPool::variants() const {
  return convertUniquePtrElementsToRawPointers(variants_);
}

Variant& DefaultVariantPool::getVariant() {
  return selectionStrategy_->selectVariant();
}

const std::string DefaultVariantPool::name() const {
  return generator_->name();
}

void DefaultVariantPool::finishQuery() { selectionStrategy_->finishQuery(); }

void DefaultVariantPool::update() {
  finishQuery();
  updateStrategy_->updatePool(*this);
  selectionStrategy_->reset(*this, false);
}

void DefaultVariantPool::swapVariants(const unsigned index1,
                                      const unsigned index2) {
  std::swap(variants_[index1], variants_[index2]);
}

void DefaultVariantPool::updateVariant(const unsigned index,
                                       std::unique_ptr<Variant> variant) {
  variants_[index] = std::move(variant);
};

void DefaultVariantPool::initialize() {
  initializationStrategy_->initializePool(*this);
  selectionStrategy_->reset(*this);
  updateStrategy_->reset();
}

void DefaultVariantPool::reset() {
  selectionStrategy_->reset(*this);
  variants_.clear();
};
}
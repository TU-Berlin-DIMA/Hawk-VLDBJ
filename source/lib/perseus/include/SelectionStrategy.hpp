#ifndef PERSEUS_SELECTIONSTRATEGY_HPP
#define PERSEUS_SELECTIONSTRATEGY_HPP

namespace perseus {

  class Variant;

  class VariantPool;

  class SelectionStrategy {
   public:
    virtual Variant& selectVariant() = 0;
    virtual void reset(const VariantPool& variantPool,
                       const bool clearPerformance = true) = 0;
    virtual void finishQuery() = 0;
    virtual ~SelectionStrategy() {}
  };
}

#endif  // PERSEUS_SELECTIONSTRATEGY_HPP

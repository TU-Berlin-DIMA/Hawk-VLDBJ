#ifndef PERSEUS_VARIANTPOOL_HPP
#define PERSEUS_VARIANTPOOL_HPP

#include <memory>
#include <string>
#include <vector>
#include "Variant.hpp"

namespace perseus {

  class VariantGenerator;

  class VariantPool {
   public:
    virtual ~VariantPool() {}

    virtual const size_t poolSize() const = 0;

    virtual Variant& getVariant() = 0;

    virtual Variant& getVariant(unsigned index) const = 0;

    virtual void removeVariant(unsigned index) = 0;

    virtual void addVariant(std::unique_ptr<Variant> variant) = 0;

    virtual void swapVariants(const unsigned index1, const unsigned index2) = 0;

    virtual void updateVariant(const unsigned index,
                               std::unique_ptr<Variant> variant) = 0;

    virtual const std::vector<Variant*> variants() const = 0;

    virtual const std::string name() const = 0;

    virtual const double reductionFactor() const = 0;

    virtual const size_t minimumSize() const = 0;

    virtual const size_t initialSize() const = 0;

    virtual const VariantGenerator& generator() const = 0;

    virtual void finishQuery() = 0;

    virtual void update() = 0;

    virtual void initialize() = 0;

    virtual void reset() = 0;

    // this should probably be moved to a specialized class
    virtual const size_t chunkSize() const = 0;
  };
}

#endif  // PERSEUS_VARIANTPOOL_HPP
